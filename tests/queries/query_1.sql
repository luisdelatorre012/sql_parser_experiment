-- https://github.com/eirkostop/SQL-Northwind-exercises/blob/master/Querying-the-database-General-Exercises-Part-1.sql
SELECT ContactName, Address
FROM Customers
WHERE CustomerID IN
	(SELECT DISTINCT CustomerID
	 FROM Orders
	 WHERE ShipVia IN
		(SELECT ShipperID
		 FROM Shippers
		 WHERE CompanyName = 'Speedy Express'))