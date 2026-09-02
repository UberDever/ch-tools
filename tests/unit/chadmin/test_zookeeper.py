import os
from types import SimpleNamespace
from typing import Any, Optional
from unittest.mock import ANY, patch

import pytest
from click.testing import CliRunner
from kazoo.exceptions import NoNodeError, NotEmptyError

from ch_tools.chadmin.cli.zookeeper_group import zookeeper_group
from ch_tools.chadmin.internal.zookeeper import delete_recursive

PATH = "/clickhouse/task_queue/ddl/query/shards/replica1:9440,replica2:9440/executed"


class FakeTransaction:
    def __init__(self, zk: "FakeZooKeeper") -> None:
        self.zk = zk
        self.paths: list[str] = []

    def delete(self, path: str) -> None:
        self.paths.append(path)

    def commit(self) -> list[Any]:
        self.zk.transaction_sizes.append(len(self.paths))
        for path in self.paths:
            if path not in self.zk.children:
                return [NoNodeError() for _ in self.paths]
            if self.zk.children[path]:
                return [NotEmptyError() for _ in self.paths]

        for path in self.paths:
            self.zk.remove_leaf(path)
        return [True for _ in self.paths]


class FakeZooKeeper:
    def __init__(self, children: dict[str, set[str]]) -> None:
        self.children = {path: set(names) for path, names in children.items()}
        self.get_children_calls: list[str] = []
        self.transaction_sizes: list[int] = []

    def exists(self, path: str) -> Any:
        if path not in self.children:
            return None
        return SimpleNamespace(children_count=len(self.children[path]))

    def get_children(self, path: str) -> list[str]:
        self.get_children_calls.append(path)
        if path not in self.children:
            raise NoNodeError
        return sorted(self.children[path])

    def transaction(self) -> FakeTransaction:
        return FakeTransaction(self)

    def delete(self, path: str, recursive: bool = False) -> None:
        if path not in self.children:
            raise NoNodeError
        if self.children[path] and not recursive:
            raise NotEmptyError
        if recursive:
            for child in list(self.children[path]):
                self.delete(os.path.join(path, child), recursive=True)
        self.remove_leaf(path)

    def remove_leaf(self, path: str) -> None:
        parent = os.path.dirname(path)
        if parent in self.children:
            self.children[parent].discard(os.path.basename(path))
        del self.children[path]


def test_large_delete_does_not_read_each_leaf() -> None:
    leaf_names = {f"leaf-{index}" for index in range(10_001)}
    zk = FakeZooKeeper(
        {
            "/root": leaf_names,
            **{f"/root/{leaf}": set() for leaf in leaf_names},
        }
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging"):
        delete_recursive(zk, ["/root"])

    assert zk.children == {}
    assert zk.get_children_calls == ["/root"]
    assert max(zk.transaction_sizes) <= 1_000


def test_large_delete_descends_only_into_nonempty_candidates() -> None:
    leaf_names = {f"leaf-{index}" for index in range(10_001)}
    zk = FakeZooKeeper(
        {
            "/root": leaf_names | {"branch"},
            "/root/branch": {"grandchild"},
            "/root/branch/grandchild": set(),
            **{f"/root/{leaf}": set() for leaf in leaf_names},
        }
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging"):
        delete_recursive(zk, ["/root"])

    assert zk.children == {}
    assert zk.get_children_calls == ["/root", "/root/branch"]


def test_bounded_scan_does_not_queue_a_large_nested_directory() -> None:
    zk = FakeZooKeeper(
        {
            "/root": {"branch"},
            "/root/branch": {"leaf-0", "leaf-1", "leaf-2"},
            "/root/branch/leaf-0": set(),
            "/root/branch/leaf-1": set(),
            "/root/branch/leaf-2": set(),
        }
    )

    with (
        patch("ch_tools.chadmin.internal.zookeeper.logging"),
        patch(
            "ch_tools.chadmin.internal.zookeeper.LARGE_RECURSIVE_DELETE_THRESHOLD",
            3,
        ),
        patch(
            "ch_tools.chadmin.internal.zookeeper.LARGE_RECURSIVE_DELETE_BATCH_SIZE",
            2,
        ),
    ):
        delete_recursive(zk, ["/root"])

    assert zk.children == {}
    assert zk.get_children_calls == [
        "/root",
        "/root/branch",
        "/root",
        "/root/branch",
    ]


class RacingZooKeeper(FakeZooKeeper):
    def __init__(self, children: dict[str, set[str]], continuous: bool) -> None:
        super().__init__(children)
        self.continuous = continuous
        self.created = 0

    def delete(self, path: str, recursive: bool = False) -> None:
        should_race = path == "/root" and not self.children[path]
        if should_race and (self.continuous or self.created == 0):
            child = f"late-{self.created}"
            self.created += 1
            self.children[path].add(child)
            self.children[f"{path}/{child}"] = set()
            raise NotEmptyError
        super().delete(path, recursive)


def test_large_delete_retries_a_finite_creation_race() -> None:
    leaf_names = {f"leaf-{index}" for index in range(10_001)}
    zk = RacingZooKeeper(
        {
            "/root": leaf_names,
            **{f"/root/{leaf}": set() for leaf in leaf_names},
        },
        continuous=False,
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging"):
        delete_recursive(zk, ["/root"])

    assert zk.children == {}
    assert zk.created == 1
    assert zk.get_children_calls == ["/root", "/root"]


def test_large_delete_aborts_after_three_stagnant_windows() -> None:
    leaf_names = {f"leaf-{index}" for index in range(10_001)}
    zk = RacingZooKeeper(
        {
            "/root": leaf_names,
            **{f"/root/{leaf}": set() for leaf in leaf_names},
        },
        continuous=True,
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging") as mock_logging:
        with pytest.raises(RuntimeError, match="does not converge"):
            delete_recursive(zk, ["/root"])

    assert zk.created == 4
    mock_logging.error.assert_called_once()


def test_small_delete_does_not_select_large_algorithm() -> None:
    zk = FakeZooKeeper(
        {
            "/root": {"leaf"},
            "/root/leaf": set(),
        }
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging") as mock_logging:
        delete_recursive(zk, ["/root"])

    assert zk.children == {}
    assert zk.get_children_calls == ["/root", "/root/leaf"]
    assert not any(
        "Using large recursive" in call.args[0]
        for call in mock_logging.info.call_args_list
    )


class AlwaysBusyRootZooKeeper(FakeZooKeeper):
    def __init__(self, children: dict[str, set[str]]) -> None:
        super().__init__(children)
        self.root_delete_attempts = 0

    def delete(self, path: str, recursive: bool = False) -> None:
        if path == "/root":
            self.root_delete_attempts += 1
            raise NotEmptyError
        super().delete(path, recursive)


def test_small_delete_does_not_retry_forever_with_a_busy_writer() -> None:
    zk = AlwaysBusyRootZooKeeper(
        {
            "/root": {"leaf"},
            "/root/leaf": set(),
        }
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging"):
        with pytest.raises(RuntimeError, match="does not converge"):
            delete_recursive(zk, ["/root"])

    assert zk.root_delete_attempts == 3


def test_large_delete_logs_algorithm_selection_and_completion() -> None:
    leaf_names = {f"leaf-{index}" for index in range(10_001)}
    zk = FakeZooKeeper(
        {
            "/root": leaf_names,
            **{f"/root/{leaf}": set() for leaf in leaf_names},
        }
    )

    with patch("ch_tools.chadmin.internal.zookeeper.logging") as mock_logging:
        delete_recursive(zk, ["/root"])

    messages = [call.args[0] for call in mock_logging.info.call_args_list]
    assert any("Using large recursive" in message for message in messages)
    assert any("completed" in message for message in messages)


class UnconfirmedDeletionZooKeeper(FakeZooKeeper):
    def __init__(self, children: dict[str, set[str]]) -> None:
        super().__init__(children)
        self.root_deleted = False

    def exists(self, path: str) -> Any:
        if path == "/root" and self.root_deleted:
            return SimpleNamespace(children_count=0)
        return super().exists(path)

    def remove_leaf(self, path: str) -> None:
        super().remove_leaf(path)
        if path == "/root":
            self.root_deleted = True


def test_delete_reports_error_if_root_absence_is_not_confirmed() -> None:
    zk = UnconfirmedDeletionZooKeeper({"/root": set()})

    with patch("ch_tools.chadmin.internal.zookeeper.logging") as mock_logging:
        with pytest.raises(RuntimeError, match="root still exists"):
            delete_recursive(zk, ["/root"])

    mock_logging.error.assert_called_once()


@pytest.mark.parametrize(
    "args,value,make_parents",
    [
        pytest.param(["create", PATH], None, False, id="no-value"),
        pytest.param(
            ["create", "--make-parents", PATH, "value"],
            "value",
            True,
            id="value-and-make-parents",
        ),
    ],
)
def test_create_command_forwards_path_value_and_make_parents(
    args: list[str], value: Optional[str], make_parents: bool
) -> None:
    with patch(
        "ch_tools.chadmin.cli.zookeeper_group.create_zk_nodes"
    ) as mock_create_zk_nodes:
        result = CliRunner().invoke(
            zookeeper_group,
            args,
            obj={"config": {"loguru": {"handlers": {}}}},
        )

    assert result.exit_code == 0, result.output
    mock_create_zk_nodes.assert_called_once_with(
        ANY, [PATH], value, make_parents=make_parents
    )


PATHS = [PATH, "/clickhouse/task_queue/ddl/query/shards/replica3/executed"]


@pytest.mark.parametrize(
    "args,command,command_args,command_kwargs",
    [
        pytest.param(
            [
                "create",
                "--path",
                PATHS[0],
                "--path",
                PATHS[1],
                "--value",
                "value",
                "--make-parents",
            ],
            "create_zk_nodes",
            (PATHS, "value"),
            {"make_parents": True},
            id="create",
        ),
        pytest.param(
            [
                "update",
                "--path",
                PATHS[0],
                "--path",
                PATHS[1],
                "--value",
                "value",
            ],
            "update_zk_nodes",
            (PATHS, "value"),
            {},
            id="update",
        ),
        pytest.param(
            ["delete", "--path", PATHS[0], "--path", PATHS[1]],
            "delete_zk_nodes",
            (PATHS,),
            {},
            id="delete",
        ),
    ],
)
def test_commands_support_repeated_path_options(
    args: list[str],
    command: str,
    command_args: tuple[Any, ...],
    command_kwargs: dict[str, Any],
) -> None:
    with patch(f"ch_tools.chadmin.cli.zookeeper_group.{command}") as mock_command:
        result = CliRunner().invoke(
            zookeeper_group,
            args,
            obj={"config": {"loguru": {"handlers": {}}}},
        )

    assert result.exit_code == 0, result.output
    mock_command.assert_called_once_with(ANY, *command_args, **command_kwargs)


def test_delete_command_rejects_multiple_paths() -> None:
    result = CliRunner().invoke(
        zookeeper_group,
        ["delete", PATH, "/clickhouse/task_queue/ddl/query/shards/replica3/executed"],
        obj={"config": {"loguru": {"handlers": {}}}},
    )

    assert result.exit_code != 0
    assert "Got unexpected extra argument" in result.output
