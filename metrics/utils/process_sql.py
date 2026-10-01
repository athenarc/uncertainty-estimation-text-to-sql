import re
from typing import List, Optional

from sqlglot import exp, parse_one


def remove_select_column_aliases(sql_query: str) -> str:
    """Removes the aliases that exist in the select clause of the given SQL query."""

    select_clause_pattern = re.compile(
        r"(SELECT\s+)(.*?)(\s+FROM\s+)", re.IGNORECASE | re.DOTALL
    )

    # Remove aliases
    def process_select_clause(match):
        select_clause = match.group(2)
        # Regex to remove column aliases that are not part of aggregate functions
        column_alias_pattern = re.compile(
            r"(\b\w+(\.\w+)?)\s+AS\s+\w+\b(?!\s*\()", re.IGNORECASE
        )
        modified_select_clause = re.sub(column_alias_pattern, r"\1", select_clause)
        return f"{match.group(1)}{modified_select_clause}{match.group(3)}"

    return re.sub(select_clause_pattern, process_select_clause, sql_query, count=1)


def get_select_attributes(sql_query: str) -> list[str]:
    """Returns a list with the attributes existing in the select clause."""
    # Regex to match and capture the content of the SELECT clause
    select_clause_pattern = re.compile(
        r"SELECT\s+(.*?)\s+FROM\s+", re.IGNORECASE | re.DOTALL
    )

    # Search for the pattern in the SQL query
    match = re.search(select_clause_pattern, sql_query)
    select_attributes = [attribute.strip() for attribute in match.group(1).split(",")]

    return select_attributes


def add_column_in_select(sql_query: str, new_column: str) -> str:
    """Updates the select clause of the query by inserting the given column."""
    # Find the select clause
    select_clause_pattern = re.compile(
        r"(SELECT\s+)(.*?)(\s+FROM\s+)", re.IGNORECASE | re.DOTALL
    )

    def add_column(match):
        select_start = match.group(1)
        select_columns = match.group(2)
        from_part = match.group(3)

        # Append the new column to the existing columns
        updated_select_columns = f"{select_columns}, {new_column}"

        # Return the updated query parts
        return f"{select_start}{updated_select_columns}{from_part}"

    # Apply the regex to the SQL query to modify the SELECT clause
    modified_query = re.sub(select_clause_pattern, add_column, sql_query, count=1)

    return modified_query


def exists_group_by(sql_query: str, dialect: str = "postgres") -> bool:
    """Returns True if there is a group by clause in the query. (Subqueries are not considered)"""
    parsed_sql = parse_one(sql_query, dialect=dialect)

    if parsed_sql is not None:
        return "group" in parsed_sql.args
    else:
        return False


def exists_order_by(sql_query: str, dialect: str = "postgres") -> bool:
    """Returns True if there is a order by clause in the query. (Subqueries are not considered)"""
    parsed_sql = parse_one(sql_query, dialect=dialect)

    if parsed_sql is not None:
        return "order" in parsed_sql.args
    else:
        return False


def get_base_tables(parsed_sql: exp.Expression) -> list[exp.Table]:
    tables = []
    if isinstance(parsed_sql.args["from"].args["this"], exp.Table):
        tables.append(parsed_sql.args["from"].args["this"])

    if "joins" in parsed_sql.args:
        for join in parsed_sql.args["joins"]:
            if isinstance(join.args["this"], exp.Table):
                tables.append(join.args["this"])
    return tables


def get_base_query_table_aliases(
    sql_query: str, dialect: str = "postgres"
) -> dict[str, str]:
    aliases = {}
    parsed_sql = parse_one(sql_query, dialect=dialect)

    base_tables = get_base_tables(parsed_sql)
    for base_table in base_tables:
        if len(base_table.alias):
            aliases[base_table.alias] = base_table.name

    return aliases


def base_table_exists(
    sql_query: str, table_name: str, dialect: str = "postgres"
) -> bool:
    parsed_sql = parse_one(sql_query, dialect=dialect)

    base_tables = get_base_tables(parsed_sql)

    for base_table in base_tables:
        if base_table.name == table_name:
            return True

    return False


def has_full_star(select_attributes: List[str]) -> bool:
    for column in select_attributes:
        if column == "*":
            return True
    return False


def has_star_with_prefix(select_attributes: List[str]) -> bool:
    for column in select_attributes:
        if ".*" in column:
            return True
    return False


def get_base_select_aliases(sql_query: str, dialect: str = "postgres") -> dict:
    """
    Returns the aliases in the base select clause of the given sql query with their corresponding value.
    When there is a function the stringify function is returned as the value of the alias.
    """

    parsed_sql = parse_one(sql_query, dialect=dialect)

    aliases_dict = {}
    # For every select expression
    for e in parsed_sql.args["expressions"]:
        # If there is an alias
        if isinstance(e, exp.Alias):
            # If this is a column remove the alias
            if isinstance(e.args["this"], exp.Column):
                column_name = e.args["this"].sql()
                aliases_dict[e.alias] = (
                    column_name if "." not in column_name else column_name.split(".")[1]
                )
            else:
                # Add as the value the string of the expression without any table name or table alias
                aliases_dict[e.alias] = (
                    re.sub(r"(\b)(\w+\.)(\w+)", r"\1\3", e.args["this"].sql())
                    .split("(")[0]
                    .lower()
                )

    return aliases_dict


def get_base_query_limit(query: str, dialect: str = "postgres") -> Optional[int]:

    parsed_sql = parse_one(query, dialect=dialect)

    if parsed_sql.args["limit"] is not None:
        return int(parsed_sql.args.get("limit").args["expression"].this)

    return None
