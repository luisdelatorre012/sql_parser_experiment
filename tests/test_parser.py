import pytest

from query_parser import (
    transform_subqueries_to_ctes,
    transform_ctes_to_subqueries,
    MultipleQueriesError,
    get_outer_tables
)


def remove_whitespace(sql: str) -> str:
    """Normalize SQL by standardizing whitespace and case."""
    # First remove all whitespace
    sql = sql.upper()

    # Add space after common punctuation
    for char in [',', '(', ')', '=']:
        sql = sql.replace(char, f"{char} ")

    # Remove extra spaces around these characters
    for char in ['(', ')', '.']:
        sql = sql.replace(f" {char} ", f"{char}")

    # Fix spacing for operators
    for op in [' = ', ' > ', ' < ', ' >= ', ' <= ', ' <> ']:
        sql = sql.replace(op.upper(), op)

    # Fix JOIN keywords spacing
    for join in ['INNER JOIN', 'LEFT JOIN', 'RIGHT JOIN', 'OUTER JOIN', 'CROSS JOIN', 'JOIN']:
        sql = sql.replace(f" {join} ", f" {join} ")

    # Collapse all whitespace
    return " ".join(sql.split())


# --- Tests for get_outer_tables() ---
@pytest.mark.parametrize("query, expected_tables", [
    (
            "SELECT t1.a, (SELECT t2.b FROM table2 t2) FROM table1 t1",
            {"TABLE1"}
    ),
    (
            "SELECT a FROM table1 JOIN table2 ON table1.id = table2.id",
            {"TABLE1", "TABLE2"}
    ),
    (
            """
            SELECT t1.a,
                   (SELECT t2.b FROM table2 t2 WHERE t2.id IN (SELECT id FROM table3))
            FROM table1 t1
            JOIN table4 t4 ON t1.id = t4.id
            WHERE t1.status = 'active'
            """,
            {"TABLE1", "TABLE4"}
    ),
])
def test_get_outer_tables(query: str, expected_tables: set) -> None:
    outer_tables = get_outer_tables(query)
    normalized_tables = {table.upper() for table in outer_tables}
    assert expected_tables.issubset(normalized_tables)


@pytest.mark.parametrize("query, expected", [
    # No subqueries - should remain unchanged
    (
            "SELECT a, b FROM table1",
            "SELECT a, b FROM table1"
    ),
    # Uncorrelated scalar subquery in SELECT
    (
            """
            SELECT a, b,
                   (SELECT MAX(c) FROM table2) as max_c
            FROM table1
            """,
            """
            WITH cte_0 AS (
                SELECT MAX(c) AS val FROM table2
            )
            SELECT a, b, cte_0.val as max_c 
            FROM table1 CROSS JOIN cte_0
            """
    ),
    # Subquery in WHERE clause
    (
            """
            SELECT a
            FROM table1
            WHERE a > (SELECT AVG(b) FROM table2)
            """,
            """
            WITH cte_0 AS (
                SELECT AVG(b) AS val FROM table2
            )
            SELECT a
            FROM table1 CROSS JOIN cte_0
            WHERE a > cte_0.val
            """
    ),
    # Subquery in JOIN clause
    (
            """
            SELECT t1.a, t2.max_val
            FROM table1 t1
            JOIN (SELECT id, MAX(val) as max_val FROM table2 GROUP BY id) t2 ON t1.id = t2.id
            """,
            """
            WITH cte_0 AS (
                SELECT id, MAX(val) as max_val FROM table2 GROUP BY id
            )
            SELECT t1.a, t2.max_val
            FROM table1 t1
            JOIN cte_0 t2 ON t1.id = t2.id
            """
    ),
    # Duplicated subqueries (should be deduplicated)
    (
            """
            SELECT a,
                   (SELECT MAX(c) FROM table2) as max_c,
                   (SELECT MAX(c) FROM table2) as another_max
            FROM table1
            """,
            """
            WITH cte_0 AS (
                SELECT MAX(c) AS val FROM table2
            )
            SELECT a, cte_0.val as max_c, cte_0.val as another_max
            FROM table1 CROSS JOIN cte_0
            """
    ),
    # Correlated subquery (should not be transformed)
    (
            """
            SELECT a, b,
                   (SELECT MAX(c) FROM table2 WHERE table2.id = table1.id) as max_c
            FROM table1
            """,
            """
            SELECT a, b,
                   (SELECT MAX(c) FROM table2 WHERE table2.id = table1.id) as max_c
            FROM table1
            """
    ),
    # Multiple different subqueries
    (
            """
            SELECT a, 
                   (SELECT MAX(c) FROM table2) as max_c,
                   (SELECT MIN(d) FROM table3) as min_d
            FROM table1
            WHERE a > (SELECT AVG(e) FROM table4)
            """,
            """
            WITH cte_0 AS (
                SELECT MAX(c) AS val FROM table2
            ),
            cte_1 AS (
                SELECT MIN(d) AS val FROM table3
            ),
            cte_2 AS (
                SELECT AVG(e) AS val FROM table4
            )
            SELECT 
                a, 
                cte_0.val as max_c, 
                cte_1.val as min_d
            FROM table1 
            CROSS JOIN cte_0 
            CROSS JOIN cte_1 
            CROSS JOIN cte_2
            WHERE a > cte_2.val
            """
    ),
])
def test_transform_subqueries_to_ctes(query, expected):
    """Test that transform_query_to_ctes produces the expected full output."""
    result = transform_subqueries_to_ctes(query)

    # Remove all whitespace for comparison
    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())
    assert normalized_result == normalized_expected


@pytest.mark.parametrize("query, expected", [
    (
            """
            SELECT a,
                   (SELECT MAX(c) FROM table2) as max_c,
                   (SELECT MAX(c) FROM table2) as another_max
            FROM table1
            """,
            """
            WITH cte_0 AS(
            SELECT 
                MAX(c) AS val
               FROM table2
            )
            SELECT a,
                   cte_0.val AS max_c,
                   cte_0.val AS another_max
            FROM table1
            CROSS JOIN cte_0
            """
    ),
])
def test_duplicate_subqueries(query: str, expected: str) -> None:
    result = transform_subqueries_to_ctes(query)

    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())

    assert normalized_result == normalized_expected


@pytest.mark.parametrize("query", [
    (
            """
            SELECT a, b,
                   (SELECT c FROM table2 UNION SELECT c FROM table3) as union_c
            FROM table1
            """
    ),
    (
            """
            SELECT a
            FROM table1
            WHERE EXISTS (SELECT 1 FROM table2 WHERE table2.id = table1.id)
            """
    ),
    (
            """
            SELECT a,
                   (SELECT AVG(val) OVER (PARTITION BY id) FROM table2) as avg_val
            FROM table1
            """
    ),
    "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) FROM table2 WHERE table2.id = table1.id)",
    "SELECT a, (SELECT AVG(val) OVER (PARTITION BY id) FROM table2) as avg_val FROM table1"
])
def test_non_transformable_subqueries(query: str) -> None:
    result = transform_subqueries_to_ctes(query)
    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(query).split())
    assert normalized_result == normalized_expected


@pytest.mark.parametrize("query", [
    "SELECT a FROM table1; SELECT b FROM table2;",
])
def test_multiple_queries_exception(query: str) -> None:
    with pytest.raises(MultipleQueriesError):
        transform_subqueries_to_ctes(query)


def test_get_outer_tables_nested() -> None:
    query = """
    SELECT t1.a,
           (SELECT t2.b FROM table2 t2 WHERE t2.id IN (SELECT id FROM table3)) as b_val
    FROM table1 t1
    JOIN table4 t4 ON t1.id = t4.id
    WHERE t1.status = 'active'
    """
    outer_tables: set = get_outer_tables(query)
    normalized_tables = {table.upper() for table in outer_tables}
    assert "TABLE1" in normalized_tables
    assert "TABLE4" in normalized_tables
    assert "TABLE2" not in normalized_tables
    assert "TABLE3" not in normalized_tables


def test_query_formatting() -> None:
    query = "SELECT a,b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)"
    result = transform_subqueries_to_ctes(query)
    # Check that line breaks exist and that the normalized output starts with a WITH or SELECT.
    assert "\n" in result
    assert remove_whitespace(result).startswith("WITH") or remove_whitespace(result).startswith("SELECT")


def test_full_output_scalar_where() -> None:
    """
    Test a query with a scalar subquery in the WHERE clause.
    Expected transformation:

    WITH cte_0 AS (
        SELECT AVG(b) AS val FROM table2
    )
    SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val
    """
    query = "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)"
    expected = (
        """
        WITH cte_0 AS (
            SELECT AVG(b) AS val FROM table2
        )
        SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val        
        """
    )
    result = transform_subqueries_to_ctes(query)
    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())
    assert normalized_result == normalized_expected


def test_full_output_duplicate_scalar() -> None:
    """
    Test a query where the same scalar subquery appears twice.
    Expected transformation:
    """
    query = """
        SELECT a,
        (SELECT MAX(c) FROM table2) as max_c,
        (SELECT MAX(c) FROM table2) as another_max
        FROM table1
        """
    expected = """
    WITH cte_0 AS (
        SELECT MAX(c) AS val FROM table2
    )
    SELECT a, cte_0.val AS max_c, cte_0.val AS another_max
    FROM table1 CROSS JOIN cte_0
    """
    result = transform_subqueries_to_ctes(query)
    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())
    assert normalized_result == normalized_expected


# --- New tests for transform_ctes_to_subqueries() --- #

def test_transform_ctes_to_subqueries_no_with() -> None:
    """
    When the query does not start with a WITH clause,
    transform_ctes_to_subqueries() should return the original query.
    """
    query = "SELECT a, b FROM table1 WHERE a > 10"
    result = transform_ctes_to_subqueries(query)
    assert remove_whitespace(result) == remove_whitespace(query)


@pytest.mark.parametrize("query, expected", [
    (
            # Basic scalar subquery transformation:
            """
            WITH cte_0 AS (
                SELECT AVG(b) AS val FROM table2
            )
            SELECT a, b FROM table1 CROSS JOIN cte_0 WHERE a > cte_0.val    
            """,
            "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS VAL FROM table2)"
    ),
])
def test_transform_ctes_to_subqueries_basic(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)

    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())

    assert normalized_result == normalized_expected


@pytest.mark.parametrize("query, expected", [
    (
            # Duplicate scalar subqueries transformed back inline:
            """
            WITH cte_0 AS (
                SELECT MAX(c) AS val FROM table2
            )
            SELECT 
                a, 
                cte_0.val AS max_c, 
                cte_0.val AS another_max 
            FROM table1 CROSS JOIN cte_0
            """,
            "SELECT a, (SELECT MAX(c) AS VAL FROM table2) AS max_c, (SELECT MAX(c) AS VAL  FROM table2) AS another_max FROM table1"
    ),
])
def test_transform_ctes_to_subqueries_duplicate(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)
    assert remove_whitespace(result) == remove_whitespace(expected)


@pytest.mark.parametrize("query, expected", [
    (
            """
            WITH cte_0 AS (SELECT AVG(b) AS val FROM table2), 
                 cte_1 AS (SELECT MAX(c) AS val FROM table3)
            SELECT 
                a, b 
            FROM table1 
            CROSS JOIN cte_0 
            CROSS JOIN cte_1
            WHERE a > cte_0.val AND b = cte_1
            """,
            "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS VAL FROM table2) AND b = (SELECT MAX(c) AS VAL FROM table3)"
    ),
])
def test_transform_ctes_to_subqueries_multiple(query: str, expected: str) -> None:
    result = transform_ctes_to_subqueries(query)
    normalized_result = ''.join(remove_whitespace(result).split())
    normalized_expected = ''.join(remove_whitespace(expected).split())

    assert normalized_result == normalized_expected
