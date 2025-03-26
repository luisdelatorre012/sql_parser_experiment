from query_parser import transform_subqueries_to_ctes, transform_ctes_to_subqueries

def main():
    """
    WITH cte_0 AS (
        SELECT MAX(c) AS val FROM table2
    )
    SELECT a, cte_0.val as max_c, cte_0.val as another_max
    FROM table1 CROSS JOIN cte_0
    """

    query = """
    WITH cte_0 AS (SELECT AVG(b) AS val FROM table2), 
         cte_1 AS (SELECT MAX(c) AS val FROM table3)
    SELECT 
        a, b 
    FROM table1 
    CROSS JOIN cte_0 
    CROSS JOIN cte_1
    WHERE a > cte_0.val AND b = cte_1
    """
    transformed_query = transform_ctes_to_subqueries(query)

    print("transformed query:")
    print(transformed_query)
    print()
    print("expected query:")
    expected_query = """
    SELECT 
        a, b 
    FROM table1 
    WHERE 
        a > (SELECT AVG(b) AS VAL FROM table2) 
        AND b = (SELECT MAX(c) AS VAL FROM table3)
"""

    print(expected_query)



if __name__ == '__main__':
    main()

