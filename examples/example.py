from query_parser import transform_subqueries_to_ctes, transform_ctes_to_subqueries


def main():
    query = """
            WITH cte_0 AS (SELECT AVG(b) AS val
                           FROM table2)
            SELECT a, b
            FROM table1
                     CROSS JOIN cte_0
            WHERE a > cte_0.val"""

    transformed_query = transform_ctes_to_subqueries(query, dialect="tsql")

    print("transformed query:")
    print(transformed_query)
    print()
    print("expected query:")
    expected_query = "SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) AS VAL FROM table2)"

    print(expected_query)


if __name__ == '__main__':
    main()
