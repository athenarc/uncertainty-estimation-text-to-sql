from typing import Dict, List, Literal, Optional, Any

VALUE_LEN_THRESHOLD = 30


def get_value_sequence(values: List) -> str:
    value_sequence = "values : "
    for i, value in enumerate(values):
        # Shorten very long values
        if type(value) == str and len(value) > VALUE_LEN_THRESHOLD:
            value = value[:VALUE_LEN_THRESHOLD] + "..."

        value_sequence += f"{value.__repr__()}"

        if i + 1 != len(values):
            # Do not add a comma after the last value
            value_sequence += " , "

    return value_sequence


def get_db_schema_sequence(
    schema: List,
    type: Literal["ddl", "compact", "m-schema"] = "ddl",
    include_pk: bool = True,
    include_fk: bool = True,
    include_notes: bool = True,
    values_num: int = 0,
    categorical_threshold: int = None,
    db_id: str = "Anonymous",
) -> str:
    """_summary_

    Args:
        schema (List): The schema list containing dictionaries of tables.
        type (Literal["ddl", "compact"], optional): The type of sequence to
            generate. Defaults to "ddl".
        include_pk (bool, optional): Whether or not to include primary key
            information . Defaults to True.
        include_fk (bool, optional): Whether or not to include foreign key
            information. Defaults to True.
        include_notes (bool, optional): Whether or not to include notes for tables
            and columns. Defaults to True.
        values_num (int, optional): The number of values to include in the schema sequence. Defaults to 0.
        categorical_threshold (int, optional): The number of values for a column to be considered categorical.
            If a column is categorical all its values will be included in the schema sequence.

    Raises:
        ValueError: If schema sequence type is not supported.

    Returns:
        str: The schema sequence string.
    """
    if type == "ddl":
        return get_ddl_schema_sequence(
            schema,
            include_pk,
            include_fk,
            include_notes,
            values_num,
            categorical_threshold,
        )
    elif type == "compact":
        return get_compact_schema_sequence(
            schema,
            include_pk,
            include_fk,
            include_notes,
            values_num,
            categorical_threshold,
        )
    elif type == "m-schema":
        return get_m_schema_sequence(
            schema,
            include_pk=include_pk,
            include_fk=include_fk,
            include_notes=include_notes,
            values_num=values_num,
            categorical_threshold=categorical_threshold,
            db_id=db_id,
        )
    else:
        raise ValueError(f"Schema sequence type {type} is not supported!")


def get_compact_schema_sequence(
    schema, include_pk, include_fk, include_notes, values_num, categorical_threshold
):
    """Create the serialized string sequence of the DB schema.

    Args:
        include_fk (bool): Whether or not to include foreign key information.
    """

    foreign_keys = []
    schema_sequence = ""

    for table in schema:
        table_name = table["table_name"]

        # Get table note if there is one
        if include_notes and table.get("note", None) is not None:
            table_note = " ( comment : " + table["note"] + " )"
        else:
            table_note = ""

        column_info_list = []
        for column in table["columns"]:
            # Get column name
            column_name = column["column"]

            additional_column_info = []

            # Get column type
            additional_column_info.append(column["data_type"])

            # Get PK indicator
            if column["is_pk"] and include_pk:
                additional_column_info.append("primary key")

            # Get foreign keys
            if len(column["foreign_keys"]) > 0:
                for fk in column["foreign_keys"]:
                    foreign_keys.append(
                        (
                            table_name,
                            column_name,
                            fk["foreign_table"],
                            fk["foreign_column"],
                        )
                    )

            # Get column note
            if include_notes and column.get("note", None) is not None:
                additional_column_info.append("comment : " + column["note"])

            # Get sample values
            if values_num != 0 and len(column["values"]) != 0:
                value_sequence = get_value_sequence(
                    column["values"][:values_num]
                    if len(column["values"]) > categorical_threshold
                    else column["values"]
                )
                additional_column_info.append(value_sequence)

            column_info_list.append(
                table_name
                + "."
                + column_name
                + " ( "
                + " | ".join(additional_column_info)
                + " )"
            )

        schema_sequence += (
            "table "
            + table_name
            + table_note
            + " , columns = [ "
            + " , ".join(column_info_list)
            + " ]\n"
        )

    if include_fk and len(foreign_keys) != 0:
        schema_sequence += "foreign keys :\n"
        for foreign_key in foreign_keys:
            schema_sequence += "{}.{} = {}.{}\n".format(
                foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3]
            )

    return schema_sequence.strip()


def get_ddl_schema_sequence(
    schema, include_pk, include_fk, include_notes, values_num, categorical_threshold
):
    foreign_keys = []
    schema_sequence = ""

    for table in schema:
        table_name = table["table_name"]

        if include_notes and table.get("note", None) is not None:
            table_note = " -- " + table["note"]
        else:
            table_note = ""

        column_info_list = []
        for column in table["columns"]:
            column_sequence = "\t"

            # Get column name
            column_name = column["column"]
            column_sequence += column_name

            # Get column type
            column_sequence += " " + column["data_type"]

            # Get PK indicator
            if column["is_pk"] and include_pk:
                column_sequence += " primary key"

            # Add comma and gather potential additional info
            column_sequence += ","
            additional_column_info = []

            # Get foreign keys
            if len(column["foreign_keys"]) > 0:
                for fk in column["foreign_keys"]:
                    foreign_keys.append(
                        (
                            table_name,
                            column_name,
                            fk["foreign_table"],
                            fk["foreign_column"],
                        )
                    )

            # Get column comment
            if include_notes and column.get("note", None) is not None:
                additional_column_info.append("comment : " + column["note"])

            # Get sample values
            if values_num != 0 and len(column["values"]) != 0:
                value_sequence = get_value_sequence(
                    column["values"][:values_num]
                    if len(column["values"]) > categorical_threshold
                    else column["values"]
                )
                additional_column_info.append(value_sequence)

            if len(additional_column_info) > 0:
                column_sequence += f" -- {', '.join(additional_column_info)}"
            column_info_list.append(column_sequence)

        schema_sequence += (
            "CREATE TABLE "
            + table_name
            + " ("
            + table_note
            + "\n"
            + "\n".join(column_info_list)
            + "\n);\n\n"
        )

    if include_fk and len(foreign_keys) != 0:
        schema_sequence += "-- Also take into account the following foreign keys :\n"
        for foreign_key in foreign_keys:
            schema_sequence += "-- {}.{} = {}.{}\n".format(
                foreign_key[0], foreign_key[1], foreign_key[2], foreign_key[3]
            )

    return schema_sequence.strip()


class MSchema:
    def __init__(self, db_id: str = "Anonymous", schema: Optional[str] = None):
        self.db_id = db_id
        self.schema = schema
        self.tables = {}
        self.foreign_keys = []

    def add_table(self, name, fields=None, comment=None):
        if fields is None:
            fields = {}
        self.tables[name] = {
            "fields": fields.copy(),
            "examples": [],
            "comment": comment,
        }

    def add_field(
        self,
        table_name: str,
        field_name: str,
        field_type: str = "",
        primary_key: bool = False,
        comment: str = "",
        examples: list = [],
    ):

        self.tables[table_name]["fields"][field_name] = {
            "type": field_type,
            "primary_key": primary_key,
            "comment": comment,
            "examples": examples.copy(),  
        }

    def add_foreign_key(self, table_name, field_name, ref_table_name, ref_field_name):
        self.foreign_keys.append(
            (table_name, field_name, ref_table_name, ref_field_name)
        )

    def get_field_type(self, field_type, simple_mode=True) -> str:
        if not simple_mode:
            return field_type
        else:
            return field_type.split("(")[0]

    def has_table(self, table_name: str) -> bool:
        return table_name in self.tables

    def has_column(self, table_name: str, field_name: str) -> bool:
        return field_name in self.tables.get(table_name, {}).get("fields", {})

    def get_field_info(self, table_name: str, field_name: str) -> Dict:
        return self.tables.get(table_name, {}).get("fields", {}).get(field_name, {})

    def single_table_mschema(
        self, table_name: str, selected_columns: List = None, example_num=3
    ) -> str:
        table_info = self.tables.get(table_name, {})
        output = []
        table_comment = table_info.get("comment", "")
        schema_prefix = f"{self.schema}." if self.schema else ""
        comment_suffix = f", {table_comment}" if table_comment else ""
        output.append(f"# Table: {schema_prefix}{table_name}{comment_suffix}")

        field_lines = []
        for field_name, field_info in table_info["fields"].items():
            if selected_columns and field_name.lower() not in selected_columns:
                continue

            raw_type = self.get_field_type(field_info["type"], simple_mode=True)
            field_line = f"({field_name}:{raw_type.upper()}"
            if field_info["comment"]:
                field_line += f", {field_info['comment'].strip()}"

            if field_info.get("primary_key", False):
                field_line += ", Primary Key"

            examples = field_info.get("examples", [])
            if examples and example_num > 0:
                examples_str = ", ".join(map(repr, examples[:example_num]))
                field_line += f", Examples: [{examples_str}]"

            field_line += ")"
            field_lines.append(field_line)

        output.append("[\n" + ",\n".join(field_lines) + "\n]")
        return "\n".join(output)

    def to_mschema(
        self,
        selected_tables: List = None,
        selected_columns: List = None,
        example_num=3,
        show_type_detail=False,
    ) -> str:
        output = [f"【DB_ID】 {self.db_id}", "【Schema】"]
        selected_tables = (
            [t.lower() for t in selected_tables] if selected_tables else None
        )
        selected_columns = (
            [c.lower() for c in selected_columns] if selected_columns else None
        )

        for table_name, table_info in self.tables.items():
            if selected_tables and table_name.lower() not in selected_tables:
                continue
            output.append(
                self.single_table_mschema(table_name, selected_columns, example_num)
            )

        if self.foreign_keys:
            output.append("【Foreign keys】")
            for fk in self.foreign_keys:
                t1, c1, t2, c2 = fk
                if (not selected_tables) or (
                    t1.lower() in selected_tables and t2.lower() in selected_tables
                ):
                    output.append(f"{t1}.{c1}={t2}.{c2}")

        return "\n".join(output)


def get_m_schema_sequence(
    schema: List,
    include_pk: bool = True,
    include_fk: bool = True,
    include_notes: bool = True,
    values_num: int = 0,
    categorical_threshold: Optional[int] = None,
    db_id: str = "Anonymous",
) -> str:
    mschema = MSchema(db_id=db_id)
    for table in schema:
        table_name = table["table_name"]
        if table_name == "nestle_nl_search_logs":
            continue
        table_note = table.get("note", "") if include_notes else ""
        mschema.add_table(table_name, comment=table_note)
        for column in table["columns"]:
            column_name = column["column"]
            data_type = column["data_type"]
            is_pk = column["is_pk"] if include_pk else False
            note = column.get("note", "") if include_notes else ""

            examples = []
            if values_num > 0:
                values = column["values"]
                if values:
                    if categorical_threshold is not None and len(values) > categorical_threshold:
                        values_subset = values[:values_num]
                    else:
                        values_subset = values.copy()

                    processed_values = []
                    for value in values_subset:
                        if isinstance(value, str) and len(value) > VALUE_LEN_THRESHOLD:
                            value = value[:VALUE_LEN_THRESHOLD] + "..."
                        processed_values.append(repr(value))  
                    examples = processed_values

            if include_fk:
                for fk in column.get("foreign_keys", []):
                    mschema.add_foreign_key(
                        table_name=table_name,
                        field_name=column_name,
                        ref_table_name=fk["foreign_table"],
                        ref_field_name=fk["foreign_column"],
                    )

            mschema.add_field(
                table_name=table_name,
                field_name=column_name,
                field_type=data_type,
                primary_key=is_pk,
                comment=note,
                examples=examples,
            )

    return mschema.to_mschema(example_num=values_num)
