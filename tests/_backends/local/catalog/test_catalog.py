
import pytest

from fenic import (
    ColumnField,
    IntegerType,
    Schema,
    Session,
    StringType,
    StructField,
    StructType,
    ToolParam,
    col,
    tool_param,
)
from fenic._backends.local.catalog import (
    DEFAULT_CATALOG_NAME,
    DEFAULT_DATABASE_NAME,
)
from fenic._backends.local.system_table_client import (
    READ_ONLY_SYSTEM_SCHEMA_NAME,
)
from fenic.core.error import (
    CatalogError,
    DatabaseAlreadyExistsError,
    DatabaseNotFoundError,
    ExecutionError,
    TableAlreadyExistsError,
    TableNotFoundError,
    ToolAlreadyExistsError,
    ToolNotFoundError,
)

NON_EXISTING_CATALOG_NAME = "non_existing_catalog"
NON_EXISTING_DATABASE_NAME = "non_existing_db"
A_DATABASE_NAME = "test_db"
ANOTHER_DATABASE_NAME = "another_existing_db"
NON_EXISTING_TABLE_NAME = "non_existing_table"
TABLE_NAME_T1 = "test_table_t1"
TABLE_NAME_T2 = "test_table_t2"
TABLE_NAME_QUALIFIED = f"{DEFAULT_DATABASE_NAME}.qualified_table"
TABLE_NAME_WITH_CATALOG = f"not_typedef_default.{DEFAULT_DATABASE_NAME}.catalog_table"
TABLE_NAME_STRUCT = "test_struct_table"
SIMPLE_TABLE_SCHEMA = Schema([ColumnField("id", IntegerType)])
SIMPLE_TABLE_SCHEMA_2 = Schema([ColumnField("name", StringType)])
STRUCT_TABLE_SCHEMA = Schema(
    [
        ColumnField(
            "s1",
            StructType(
                [
                    StructField("i", IntegerType),
                    StructField("j", IntegerType),
                ]
            ),
        )
    ]
)


def test_does_catalog_exist(local_session: Session):
    assert local_session.catalog.does_catalog_exist(DEFAULT_CATALOG_NAME)
    assert not local_session.catalog.does_catalog_exist(NON_EXISTING_CATALOG_NAME)
    assert local_session.catalog.does_catalog_exist(DEFAULT_CATALOG_NAME.upper())

def test_get_current_catalog(local_session: Session):
    assert local_session.catalog.get_current_catalog() == DEFAULT_CATALOG_NAME


def test_set_current_catalog(local_session: Session):
    with pytest.raises(
        CatalogError,
        match="Invalid catalog name 'non_existing_catalog'. Only the default catalog 'typedef_default' is supported in local execution mode.",
    ):
        local_session.catalog.set_current_catalog(NON_EXISTING_CATALOG_NAME)
    # No actual change expected for the default catalog
    local_session.catalog.set_current_catalog(DEFAULT_CATALOG_NAME)
    assert local_session.catalog.get_current_catalog() == DEFAULT_CATALOG_NAME


def test_list_catalogs(local_session: Session):
    assert local_session.catalog.list_catalogs() == [DEFAULT_CATALOG_NAME]


def test_does_database_exist(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(NON_EXISTING_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME.upper())
    assert not local_session.catalog.does_database_exist(NON_EXISTING_DATABASE_NAME.upper())


def test_create_database(local_session: Session):
    assert local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.create_database(
        A_DATABASE_NAME, ignore_if_exists=True
    )
    assert not local_session.catalog.create_database(
        A_DATABASE_NAME.upper(), ignore_if_exists=True
    )
    with pytest.raises(
        DatabaseAlreadyExistsError, match=f"Database '{A_DATABASE_NAME}' already exists"
    ):
        local_session.catalog.create_database(A_DATABASE_NAME, ignore_if_exists=False)
    with pytest.raises(
        DatabaseAlreadyExistsError, match=f"Database '{A_DATABASE_NAME.upper()}' already exists"
    ):
        local_session.catalog.create_database(A_DATABASE_NAME.upper(), ignore_if_exists=False)


def test_drop_database(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    assert local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert local_session.catalog.drop_database(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(A_DATABASE_NAME)
    assert not local_session.catalog.does_database_exist(A_DATABASE_NAME.upper())
    assert not local_session.catalog.drop_database(NON_EXISTING_DATABASE_NAME)
    assert not local_session.catalog.drop_database(
        NON_EXISTING_DATABASE_NAME, ignore_if_not_exists=True
    )
    with pytest.raises(
        DatabaseNotFoundError,
        match=f"Database '{NON_EXISTING_DATABASE_NAME}' does not exist",
    ):
        local_session.catalog.drop_database(
            NON_EXISTING_DATABASE_NAME, ignore_if_not_exists=False
        )

    local_session.catalog.create_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(
        CatalogError,
        match=f"Cannot drop the current database '{DEFAULT_DATABASE_NAME}'. Switch to another database first.",
    ):
        local_session.catalog.drop_database(DEFAULT_DATABASE_NAME)

    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(CatalogError):
        local_session.catalog.drop_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=True)
    assert not local_session.catalog.does_database_exist(ANOTHER_DATABASE_NAME)


    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    local_session.create_dataframe({"a": [1, 2, 3]}).write.save_as_view("df1")
    assert local_session.catalog.does_view_exist("df1")
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    with pytest.raises(CatalogError, match="Cannot drop database 'another_existing_db' because it contains views. Use CASCADE to drop the database and all its views."):
        local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=False)
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    assert local_session.catalog.does_view_exist("df1")
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    local_session.catalog.drop_database(ANOTHER_DATABASE_NAME, cascade=True)
    assert not local_session.catalog.does_database_exist(ANOTHER_DATABASE_NAME)

def test_list_databases(local_session: Session):
    local_session.catalog.create_database(A_DATABASE_NAME)
    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)
    databases = local_session.catalog.list_databases()
    assert A_DATABASE_NAME in databases
    assert ANOTHER_DATABASE_NAME in databases
    assert DEFAULT_DATABASE_NAME in databases
    assert READ_ONLY_SYSTEM_SCHEMA_NAME in databases  # Read-only system database
    assert len(databases) == 4


def test_read_only_system_database_protection(local_session: Session):
    """Test that the fenic_system database cannot be dropped."""
    with pytest.raises(CatalogError, match="Cannot drop read-only system database"):
        local_session.catalog.drop_database(READ_ONLY_SYSTEM_SCHEMA_NAME)
    
    # Test that tables in fenic_system cannot be dropped
    with pytest.raises(CatalogError, match="Cannot drop table 'fenic_system.metrics' from read-only system database"):
        local_session.catalog.drop_table(f"{READ_ONLY_SYSTEM_SCHEMA_NAME}.metrics")
    
    # Test that we cannot create tables in fenic_system
    with pytest.raises(CatalogError, match="Cannot create table 'fenic_system.test_table' in read-only system database"):
        local_session.catalog.create_table(f"{READ_ONLY_SYSTEM_SCHEMA_NAME}.test_table", SIMPLE_TABLE_SCHEMA)
    
    # Test that we cannot write to tables in fenic_system
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    with pytest.raises(ExecutionError, match="Cannot write to table 'fenic_system.test_table' in read-only system database"):
        df.write.save_as_table("fenic_system.test_table")


def test_get_and_set_current_database(local_session: Session):
    # create databases
    local_session.catalog.create_database(A_DATABASE_NAME)
    local_session.catalog.create_database(ANOTHER_DATABASE_NAME)

    local_session.catalog.set_current_database(A_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == A_DATABASE_NAME
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == DEFAULT_DATABASE_NAME
    local_session.catalog.set_current_database(ANOTHER_DATABASE_NAME)
    assert local_session.catalog.get_current_database() == ANOTHER_DATABASE_NAME


def test_does_table_exist(local_session: Session):
    # create table
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    # assert table exists
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}"
    )
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{TABLE_NAME_T1}"
    )
    assert local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{TABLE_NAME_T1}".upper()
    )
    assert not local_session.catalog.does_table_exist(NON_EXISTING_TABLE_NAME)
    assert not local_session.catalog.does_table_exist(
        f"{DEFAULT_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}"
    )

def test_does_view_exist(local_session: Session):
    local_session.catalog.set_current_database(DEFAULT_DATABASE_NAME)

    view_name = "df1"
    non_existing_view_name = "df2"

    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view(view_name)

    assert local_session.catalog.does_view_exist(view_name)
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME}.{view_name}"
    )
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{view_name}"
    )
    assert local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME.upper()}.{view_name}".upper()
    )
    assert not local_session.catalog.does_view_exist(non_existing_view_name)
    assert not local_session.catalog.does_view_exist(
        f"{DEFAULT_DATABASE_NAME}.{non_existing_view_name}"
    )

def test_list_tables(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    local_session.catalog.create_table(TABLE_NAME_T2, SIMPLE_TABLE_SCHEMA_2)
    tables = local_session.catalog.list_tables()
    assert TABLE_NAME_T1 in tables
    assert TABLE_NAME_T2 in tables

def test_list_views(local_session: Session):
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view("df1")

    df2 = local_session.create_dataframe({"b": [1, 2, 3]})
    df2.write.save_as_view("df2")

    views = local_session.catalog.list_views()
    assert "df1" in views
    assert "df2" in views

def test_describe_view(local_session: Session):
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view("df1")
    view_df1 = local_session._session_state.catalog.get_view_plan("df1")
    assert view_df1.schema().column_names() == ["a"]

def test_describe_table(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    table_metadata = local_session.catalog.describe_table(TABLE_NAME_T1)
    schema = table_metadata.schema
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "id"
    assert schema.column_fields[0].data_type == IntegerType

    table_metadata = local_session.catalog.describe_table(
        f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}"
    )
    schema = table_metadata.schema
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "id"
    assert schema.column_fields[0].data_type == IntegerType

    with pytest.raises(
        TableNotFoundError,
        match=f"Table 'typedef_default.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.describe_table(NON_EXISTING_TABLE_NAME)
    with pytest.raises(
        TableNotFoundError,
        match=f"Table '{A_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.describe_table(
            f"{A_DATABASE_NAME}.{NON_EXISTING_TABLE_NAME}"
        )


def test_describe_table_struct(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_STRUCT, STRUCT_TABLE_SCHEMA)
    table_metadata = local_session.catalog.describe_table(TABLE_NAME_STRUCT)
    schema = table_metadata.schema
    assert len(schema.column_fields) == 1
    assert schema.column_fields[0].name == "s1"
    assert isinstance(schema.column_fields[0].data_type, StructType)
    inner_schema = table_metadata.schema.column_fields[0].data_type
    assert len(inner_schema.struct_fields) == 2
    assert inner_schema.struct_fields[0].name == "i"
    assert inner_schema.struct_fields[0].data_type == IntegerType
    assert inner_schema.struct_fields[1].name == "j"
    assert inner_schema.struct_fields[1].data_type == IntegerType


def test_drop_table(local_session: Session):
    local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.drop_table(TABLE_NAME_T1)
    assert not local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.drop_table(f"{DEFAULT_DATABASE_NAME}.{TABLE_NAME_T1}")
    assert not local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert not local_session.catalog.drop_table(NON_EXISTING_TABLE_NAME)
    assert not local_session.catalog.drop_table(
        NON_EXISTING_TABLE_NAME, ignore_if_not_exists=True
    )
    with pytest.raises(
        TableNotFoundError,
        match=f"Table 'typedef_default.{NON_EXISTING_TABLE_NAME}' does not exist",
    ):
        local_session.catalog.drop_table(
            NON_EXISTING_TABLE_NAME, ignore_if_not_exists=False
        )

def test_drop_view(local_session: Session):
    view_name_1 = "df1"
    view_name_2 = "df2"
    df1 = local_session.create_dataframe({"a": [1, 2, 3]})
    df1.write.save_as_view(view_name_1)

    assert local_session.catalog.does_view_exist(view_name_1)
    assert local_session.catalog.drop_view(view_name_1)
    assert not local_session.catalog.does_view_exist(view_name_1)

    df1.write.save_as_view(view_name_2)
    assert local_session.catalog.does_view_exist(view_name_2)
    assert local_session.catalog.drop_view(f"{DEFAULT_DATABASE_NAME}.{view_name_2}")
    assert not local_session.catalog.does_view_exist(view_name_2)

    with pytest.raises(
        TableNotFoundError,
        match="Table 'typedef_default.df3' does not exist",
    ):
        local_session.catalog.drop_view(
            "typedef_default.df3", ignore_if_not_exists=False
        )


def test_save_as_table_with_description_and_metadata(local_session: Session):
    table_name = "meta_table_desc"
    df = local_session.create_dataframe({"a": [1, 2, 3]})
    # Save with description
    df.write.save_as_table(table_name, mode="overwrite")
    local_session.catalog.set_table_description(table_name, "table desc")

    meta = local_session.catalog.describe_table(table_name)
    assert meta.description == "table desc"
    assert meta.schema.column_names() == ["a"]

def test_set_description_for_non_existing_table(local_session: Session):
    table_name = "non_existing_table"
    with pytest.raises(TableNotFoundError, match=f"Table '{DEFAULT_DATABASE_NAME}.{table_name}' does not exist"):
        local_session.catalog.set_table_description(table_name, "table desc")

def test_set_description_for_non_existing_view(local_session: Session):
    view_name = "non_existing_view"
    with pytest.raises(TableNotFoundError, match=f"Table '{DEFAULT_DATABASE_NAME}.{view_name}' does not exist"):
        local_session.catalog.set_view_description(view_name, "view desc")

def test_save_as_view_with_description_and_metadata(local_session: Session):
    view_name = "meta_view_desc"
    df = local_session.create_dataframe({"b": [1, 2, 3]})
    # Save with description
    df.write.save_as_view(view_name, description="view desc")

    vmeta = local_session.catalog.describe_view(view_name)
    assert vmeta.description == "view desc"
    assert vmeta.schema.column_names() == ["b"]

    local_session.catalog.set_view_description(view_name, "updated_view_desc")
    meta = local_session.catalog.describe_view(view_name)
    assert meta.description == "updated_view_desc"


def test_get_table_metadata_without_description(local_session: Session):
    tbl = "meta_table_no_desc"
    local_session.catalog.create_table(tbl, SIMPLE_TABLE_SCHEMA)
    meta = local_session.catalog.describe_table(tbl)
    assert meta.description is None
    assert meta.schema.column_names() == ["id"]


def test_set_table_description_updates_metadata(local_session: Session):
    tbl = "meta_table_set_desc"
    local_session.catalog.create_table(tbl, SIMPLE_TABLE_SCHEMA, description="table desc")
    local_session.catalog.set_table_description(tbl, "updated desc")
    meta = local_session.catalog.describe_table(tbl)
    assert meta.description == "updated desc"
    assert meta.schema.column_names() == ["id"]
    local_session.catalog.set_table_description(tbl, None)
    meta = local_session.catalog.describe_table(tbl)
    assert meta.description is None
    assert meta.schema.column_names() == ["id"]

def test_create_table(local_session: Session):
    assert local_session.catalog.create_table(TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_T1)
    assert not local_session.catalog.create_table(
        TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA, ignore_if_exists=True
    )
    with pytest.raises(
        TableAlreadyExistsError,
        match=f"Table 'typedef_default.{TABLE_NAME_T1}' already exists",
    ):
        local_session.catalog.create_table(
            TABLE_NAME_T1, SIMPLE_TABLE_SCHEMA, ignore_if_exists=False
        )

    assert local_session.catalog.create_table(TABLE_NAME_QUALIFIED, SIMPLE_TABLE_SCHEMA)
    assert local_session.catalog.does_table_exist(TABLE_NAME_QUALIFIED)

    with pytest.raises(
        CatalogError,
        match="Invalid catalog name 'not_typedef_default'",
    ):
        local_session.catalog.create_table(TABLE_NAME_WITH_CATALOG, SIMPLE_TABLE_SCHEMA)


def test_create_tool(local_session: Session):
    df = local_session.create_dataframe({"city": ["SF", "SEA"], "age": [25, 30]})
    parameterized = df.filter(
        (col("city") == tool_param("city_name", StringType))
        & (col("age") >= tool_param("min_age", IntegerType))
    )

    created = local_session.catalog.create_tool(
        tool_name="users_by_city",
        tool_description="Filter users by city and min age",
        tool_query=parameterized,
        tool_params=[
            ToolParam(name="city_name", description="City name"),
            ToolParam(name="min_age", description="Minimum age"),
        ],
        result_limit=10,
        ignore_if_exists=False,
    )
    assert created is True

    # Creating again with ignore_if_exists=True should return False
    assert (
        local_session.catalog.create_tool(
            tool_name="users_by_city",
            tool_description="Filter users by city and min age",
            tool_query=parameterized,
            tool_params=[
                ToolParam(name="city_name", description="City name"),
                ToolParam(name="min_age", description="Minimum age"),
            ],
            result_limit=10,
            ignore_if_exists=True,
        )
        is False
    )

    # Creating again with ignore_if_exists=False should raise
    with pytest.raises(ToolAlreadyExistsError, match="Tool 'users_by_city' already exists"):
        local_session.catalog.create_tool(
            tool_name="users_by_city",
            tool_description="Filter users by city and min age",
            tool_query=parameterized,
            tool_params=[
                ToolParam(name="city_name", description="City name"),
                ToolParam(name="min_age", description="Minimum age"),
            ],
            result_limit=10,
            ignore_if_exists=False,
        )


def test_describe_tool(local_session: Session):
    df = local_session.create_dataframe({"city": ["SF"], "age": [10]})
    parameterized = df.filter(col("city") == tool_param("city_name", StringType))

    local_session.catalog.create_tool(
        tool_name="tool_desc",
        tool_description="Desc test",
        tool_query=parameterized,
        tool_params=[ToolParam(name="city_name", description="City name")],
        result_limit=7,
        ignore_if_exists=False,
    )

    tool = local_session.catalog.describe_tool("tool_desc")
    assert tool.name == "tool_desc"
    assert [p.name for p in tool.params] == ["city_name"]
    assert tool.max_result_limit == 7


def test_list_tools(local_session: Session):
    df = local_session.create_dataframe({"city": ["SF", "SEA"]})
    p = df.filter(col("city") == tool_param("city_name", StringType))

    local_session.catalog.create_tool(
        tool_name="tool_a",
        tool_description="A",
        tool_query=p,
        tool_params=[ToolParam(name="city_name", description="City name")],
        result_limit=5,
        ignore_if_exists=False,
    )
    local_session.catalog.create_tool(
        tool_name="tool_b",
        tool_description="B",
        tool_query=p,
        tool_params=[ToolParam(name="city_name", description="City name")],
        result_limit=5,
        ignore_if_exists=False,
    )

    tools = local_session.catalog.list_tools()
    names = {t.name for t in tools}
    assert {"tool_a", "tool_b"} == names


def test_drop_tool(local_session: Session):
    df = local_session.create_dataframe({"city": ["SF"], "age": [10]})
    parameterized = df.filter(col("city") == tool_param("city_name", StringType))

    local_session.catalog.create_tool(
        tool_name="tool_to_drop",
        tool_description="",
        tool_query=parameterized,
        tool_params=[ToolParam(name="city_name", description="City name")],
        result_limit=3,
        ignore_if_exists=False,
    )

    assert local_session.catalog.drop_tool("tool_to_drop") is True
    assert local_session.catalog.drop_tool("tool_to_drop", ignore_if_not_exists=True) is False

    with pytest.raises(ToolNotFoundError, match="Tool 'tool_to_drop' does not exist"):
        local_session.catalog.describe_tool("tool_to_drop")

    with pytest.raises(ToolNotFoundError, match="Tool 'tool_to_drop' does not exist"):
        local_session.catalog.drop_tool("tool_to_drop", ignore_if_not_exists=False)
