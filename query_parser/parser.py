import sqlparse
from sqlparse.sql import Parenthesis, Token
from sqlparse.tokens import DML
import re


class MultipleQueriesError(Exception):
    """Raised when multiple SQL queries are detected."""
    pass


def remove_parenthesized_content(query: str) -> str:
    result = []
    level: int = 0
    for char in query:
        if char == '(':
            level += 1
        elif char == ')':
            if level > 0:
                level -= 1
        else:
            if level == 0:
                result.append(char)
    return ''.join(result)


def get_outer_tables(query: str) -> set[str]:
    query_without_subqueries = remove_parenthesized_content(query)
    pattern = r'\b(?:FROM|JOIN)\s+([^\s,;]+)'
    tables = re.findall(pattern, query_without_subqueries, flags=re.IGNORECASE)
    return set(tables)


def find_subqueries(token_list: list[Token]) -> list[Token]:
    subqueries = []
    for token in token_list:
        if isinstance(token, Parenthesis):
            if any(t.ttype is DML and t.value.upper() == 'SELECT' for t in token.tokens):
                subqueries.append(token)
        elif token.is_group:  # is_group is a boolean property.
            subqueries.extend(find_subqueries(token.tokens))
    return subqueries


def is_correlated_subquery(subquery_str: str, outer_query: str) -> bool:
    outer_tables = get_outer_tables(outer_query)
    for table in outer_tables:
        if table in subquery_str:
            return True
    return False


def is_transformable_subquery(inner_subquery: str, outer_query: str) -> bool:
    if is_correlated_subquery(inner_subquery, outer_query):
        return False
    if re.search(r'\b(UNION|INTERSECT|EXCEPT)\b', inner_subquery, flags=re.IGNORECASE):
        return False
    if re.search(r'\b(EXISTS|NOT\s+EXISTS)\b', inner_subquery, flags=re.IGNORECASE):
        return False
    if re.search(r'\bOVER\s*\(', inner_subquery, flags=re.IGNORECASE):
        return False
    return True


def parse_single_query(query: str):
    """Parse the query and ensure only a single SQL statement is present."""
    parsed = sqlparse.parse(query)
    if not parsed:
        return None
    if len(parsed) != 1:
        raise MultipleQueriesError("Multiple queries found; only a single query is supported.")
    return parsed[0]


def extract_inner_subquery(subquery_text: str) -> str:
    """
    Given the text of a subquery, strip the outer parentheses if they exist.
    """
    if subquery_text.startswith('(') and subquery_text.endswith(')'):
        return subquery_text[1:-1].strip()
    return subquery_text


def normalize_subquery(subquery: str) -> str:
    """Normalize the subquery string by collapsing all whitespace to a single space."""
    return " ".join(subquery.split())


def is_scalar_subquery(subquery: str) -> bool:
    """
    Determine if a subquery is scalar by checking for aggregate functions.
    (This heuristic is based on the presence of AVG, SUM, COUNT, MIN, or MAX.)
    """
    return bool(re.search(r'\b(AVG|SUM|COUNT|MIN|MAX)\(', subquery, flags=re.IGNORECASE))


def create_cte_definition(inner_subquery: str, alias: str) -> str:
    """
    Create the CTE definition string. If the subquery is scalar, modify the SELECT clause
    to include an alias "val" so that the outer query can reference it as {alias}.val.
    If an aggregate function already has an alias, that alias is preserved.
    """
    if is_scalar_subquery(inner_subquery):
        # Check if there's already an alias in the SELECT clause
        select_pattern = r'SELECT\s+(.*?)(?:\s+FROM\s+|$)'
        select_match = re.search(select_pattern, inner_subquery, flags=re.IGNORECASE | re.DOTALL)

        if select_match and not re.search(r'\bAS\s+\w+', select_match.group(1), flags=re.IGNORECASE):
            # No alias found, add "AS val" before FROM clause
            new_inner_subquery = re.sub(
                r'(SELECT\s+.*?)(\s+FROM\s+)',
                r'\1 AS val\2',
                inner_subquery,
                flags=re.IGNORECASE | re.DOTALL
            )
            return f"{alias} AS ({new_inner_subquery})"

        # Alias already exists, use the subquery as is
        return f"{alias} AS ({inner_subquery})"
    else:
        return f"{alias} AS ({inner_subquery})"


def update_main_query_for_scalar_subqueries(main_query: str, uncorrelated_subqueries: dict[str, str]) -> str:
    """
    For each scalar subquery, update the main query:
      - Replace unqualified references to the alias with a qualified version (alias.val).
      - Insert a CROSS JOIN with the corresponding CTE for scalar subqueries not already in JOIN.
    """
    # First, replace all scalar references with qualified versions
    for normalized, alias in uncorrelated_subqueries.items():
        if re.search(r'\b(AVG|SUM|COUNT|MIN|MAX)\(', normalized, flags=re.IGNORECASE):
            # Don't modify aliases that appear in JOIN clauses
            join_pattern = rf'(?i)JOIN\s+{alias}\b\s+(?:AS\s+)?(\w+)'
            join_matches = re.finditer(join_pattern, main_query)

            # Store the positions of JOIN aliases to avoid modifying them
            join_positions = []
            for match in join_matches:
                start, end = match.span(0)
                join_positions.append((start, end))

            # Replace alias references with qualified ones (i.e. alias.val)
            # but only for references not in JOIN clauses
            pattern = rf'(?<!\.)\b{alias}\b'
            positions = []
            for match in re.finditer(pattern, main_query):
                start, end = match.span(0)
                # Check if this match is within a JOIN clause
                in_join = any(pos[0] <= start <= pos[1] for pos in join_positions)
                if not in_join:
                    positions.append((start, end))

            # Replace from end to start to avoid position shifts
            positions.sort(reverse=True)
            for start, end in positions:
                before = main_query[:start]
                after = main_query[end:]
                main_query = before + f"{alias}.val" + after

    # Now collect all CTEs that need cross joins
    ctes_to_join = []
    for normalized, alias in uncorrelated_subqueries.items():
        if re.search(r'\b(AVG|SUM|COUNT|MIN|MAX)\(', normalized, flags=re.IGNORECASE):
            # Only add CROSS JOIN if the CTE is not already part of a JOIN
            if not re.search(rf'(?i)JOIN\s+{alias}\b', main_query):
                ctes_to_join.append(alias)

    # Add all cross joins in proper order (matching CTE declaration order)
    if ctes_to_join:
        from_pattern = r'(FROM\s+\S+)'
        from_match = re.search(from_pattern, main_query)
        if from_match:
            join_clause = " " + " ".join(f"CROSS JOIN {alias}" for alias in ctes_to_join)
            before = main_query[:from_match.end()]
            after = main_query[from_match.end():]
            main_query = before + join_clause + after

    return main_query


def transform_subqueries_to_ctes(query: str) -> str:
    """
    Transform the given SQL query by extracting uncorrelated and transformable subqueries
    and replacing them with CTE references. For scalar subqueries (detected heuristically via
    aggregate functions), the CTE definition is modified so that the SELECT clause includes a
    column alias 'val'. The outer query gets a CROSS JOIN with the CTE and uses the qualified
    reference (e.g. cte_0.val) in predicate contexts.

    Raises:
        MultipleQueriesError: If more than one SQL statement is detected.
    """
    statement = parse_single_query(query)
    if statement is None:
        return query

    # Assume find_subqueries is defined elsewhere.
    subqueries = find_subqueries(statement.tokens)

    cte_definitions = []
    transformed_query = query
    uncorrelated_subqueries: dict[str, str] = {}
    alias_counter = 0

    # Process each subquery to see if it should be transformed into a CTE.
    for subquery in subqueries:
        subquery_text = str(subquery).strip()
        inner_subquery = extract_inner_subquery(subquery_text)
        normalized = normalize_subquery(inner_subquery)

        if not is_transformable_subquery(inner_subquery, query):
            continue

        if normalized in uncorrelated_subqueries:
            alias = uncorrelated_subqueries[normalized]
        else:
            alias = f"cte_{alias_counter}"
            alias_counter += 1
            uncorrelated_subqueries[normalized] = alias
            cte_definition = create_cte_definition(inner_subquery, alias)
            cte_definitions.append(cte_definition)

        # Check if this subquery is part of a JOIN clause
        join_pattern = rf'(?i)JOIN\s+{re.escape(subquery_text)}\s+(?:AS\s+)?(\w+)'
        join_match = re.search(join_pattern, transformed_query)

        if join_match:
            # If subquery is used in JOIN, preserve the original join semantics
            table_alias = join_match.group(1)
            # Replace just the subquery part, preserving the JOIN and any alias
            transformed_query = transformed_query.replace(subquery_text, alias)
        else:
            # For other contexts (WHERE, SELECT, etc.), replace the subquery directly
            transformed_query = transformed_query.replace(subquery_text, alias)

    # If any CTEs were created, build the WITH clause and update scalar subquery references.
    if cte_definitions:
        with_clause = "WITH " + ",\n     ".join(cte_definitions)
        main_query = update_main_query_for_scalar_subqueries(transformed_query, uncorrelated_subqueries)
        final_query = with_clause + "\n" + main_query
    else:
        final_query = transformed_query

    formatted_query = sqlparse.format(final_query, reindent=True, keyword_case='upper')
    return formatted_query


# --- New functionality: Transform CTEs back into inline subqueries --- #

def split_cte_definitions(cte_defs: str) -> list[str]:
    """
    Split the CTE definitions (as found in the WITH clause) into individual definitions,
    taking care not to split on commas within nested parentheses.
    """
    definitions = []
    current = []
    level = 0
    for char in cte_defs:
        if char == '(':
            level += 1
            current.append(char)
        elif char == ')':
            level -= 1
            current.append(char)
        elif char == ',' and level == 0:
            definitions.append(''.join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        definitions.append(''.join(current).strip())
    return definitions


def extract_cte_definitions(query: str) -> tuple[str, str]:
    """
    Given a query starting with WITH, extract the CTE definitions part and the main query.
    This function reads character-by-character and uses parenthesis balancing to determine
    where the CTE definitions end and the main query begins.
    Returns a tuple: (cte_definitions_str, main_query).
    """
    # Remove the initial "WITH" keyword.
    remaining = re.sub(r"(?is)^\s*WITH\s+", "", query, count=1)
    cte_part = []
    i = 0
    depth = 0
    length = len(remaining)
    main_query_start = None
    while i < length:
        # At depth 0, check if the remaining text begins with a DML keyword.
        if depth == 0:
            m = re.match(r"(?i)^(SELECT|INSERT|UPDATE|DELETE)\b", remaining[i:])
            if m:
                main_query_start = i
                break
        char = remaining[i]
        cte_part.append(char)
        if char == '(':
            depth += 1
        elif char == ')':
            depth -= 1
        i += 1
    if main_query_start is None:
        return ("".join(cte_part).strip(), "")
    cte_defs_str = "".join(cte_part).strip()
    main_query = remaining[main_query_start:].strip()
    return (cte_defs_str, main_query)


def transform_ctes_to_subqueries(query: str) -> str:
    """
    Reverse the transformation done by transform_subqueries_to_ctes().
    If the query starts with a WITH clause defining CTEs, this function extracts those
    definitions and substitutes the CTE references in the main query with the original subqueries.

    It also removes any CROSS JOINs inserted for scalar subqueries.
    """
    # Only process if the query starts with WITH.
    if not re.match(r"(?is)^\s*WITH\s+", query):
        return query

    # Extract the CTE definitions block and the main query using the custom parser.
    cte_defs_str, main_query = extract_cte_definitions(query)
    if not main_query:
        return query

    # Split individual CTE definitions.
    cte_def_list = split_cte_definitions(cte_defs_str)
    alias_to_subquery = {}
    # Extract the alias and subquery text from each definition.
    # Each definition is assumed to be of the form: alias AS ( subquery )
    for cte_def in cte_def_list:
        m_cte = re.match(r"(?is)^\s*(\S+)\s+AS\s*\((.*)\)\s*$", cte_def, flags=re.DOTALL)
        if m_cte:
            alias = m_cte.group(1).strip()
            subquery_text = m_cte.group(2).strip()
            alias_to_subquery[alias] = subquery_text

    # For each CTE alias, remove any inserted CROSS JOINs and replace alias references in the main query.
    for alias, subquery in alias_to_subquery.items():
        # Remove any CROSS JOIN clauses for this alias.
        main_query = re.sub(rf"(?is)\s+CROSS\s+JOIN\s+{alias}\b", " ", main_query)
        # First, replace qualified references (e.g. cte_0.val) with the inline subquery.
        main_query = re.sub(rf"(?<!\w){alias}\.val\b", f"({subquery})", main_query)
        # Then, replace unqualified references to the alias.
        main_query = re.sub(rf"(?<!\w){alias}\b", f"({subquery})", main_query)

    # Optionally, format the final query.
    final_query = sqlparse.format(main_query, reindent=True, keyword_case='upper')
    return final_query


# Example usage:
if __name__ == "__main__":
    original_query = """
    SELECT a, 
           (SELECT MAX(c) FROM table2) as max_c,
           (SELECT MIN(d) FROM table3) as min_d
    FROM table1
    WHERE a > (SELECT AVG(e) FROM table4)
    """

    # Transform subqueries to CTEs.
    cte_query = transform_subqueries_to_ctes(original_query)
    print("Query with CTEs:")
    print(cte_query)
    print("\n-----------------\n")

    # Reverse: Transform CTEs back to inline subqueries.
    reversed_query = transform_ctes_to_subqueries(cte_query)
    print("Reversed query (CTEs to subqueries):")
    print(reversed_query)
