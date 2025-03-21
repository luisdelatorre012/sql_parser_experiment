SELECT ProductName
FROM Products
WHERE CategoryID IN
	(SELECT CategoryID
	 FROM Categories
	 WHERE CategoryName = 'Condiments');
