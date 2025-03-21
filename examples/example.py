from query_parser import transform_query_to_ctes


def main():
    query = """
SELECT a, b FROM table1 WHERE a > (SELECT AVG(b) FROM table2)
    """
    transformed_query = transform_query_to_ctes(query)
    print(transformed_query)


if __name__ == '__main__':
    main()

