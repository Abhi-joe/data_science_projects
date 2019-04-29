#imports for web-scraping
import requests
from bs4 import BeautifulSoup as bs
#import for HTML objects display
from IPython.core.display import display, HTML

html_string="""
<!DOCTYPE html>
<html>
    <head>
        <title>Data Science test</title>
    </head>
    <body style="align-content: center">
        <h1 style="color:darkmagenta">Doing Data Science with Python</h1>
        <p id="author" style="color: darkslateblue">author: Abhirup Debnath</p>
        <p id="description" style="color: darkslateblue">description: This is a test demo for web-scraping techniques</p>
        <h3 style="color: darkviolet">Production Data(Health Items)</h3>
        <table id="module" style="width: 50%, align-self:auto" border="10">
            <tr> 
                <th>Product</th>
                <th>Production(in thousands) 2012</th>
                <th>Exports(in thousands) 2012</th>
            </tr>
            <tr>
                <td>Fish Oil</td>
                <td>90</td>
                <td>18</td>
            </tr>
            <tr>
                <td>Whey Protien</td>
                <td>64</td>
                <td>8</td>
            </tr>
            <tr>
                <td>BCAA</td>
                <td>72</td>
                <td>32</td>
            </tr>
            <tr>
                <td>Creatine</td>
                <td>81</td>
                <td>27</td>
            </tr>
            <tr>
                <td>Fat burners</td>
                <td>128</td>
                <td>40</td>
            </tr>
        </table>
    </body>
</html>
"""

#for displaying HTML objects
display(HTML(html_string)) 

ps= bs(html_string)

print(ps)

body= ps.find(name="body")
print(body)

title=body.find(name="h1").text

print(title)

print(body.find(name="p"))
print(body.findAll(name="p"))

for p in body.findAll(name="p"):
    print(p.text)

print(body.find(name="p", attrs={"id":"author"}))
print(body.find(name="p", attrs={"id":"description"}))

#Data extraction through web-scrapping

body_data=bs(html_string).find(name="body")

table_data=body_data.find(name="table", attrs={"id":"module"})

print(body_data.find(name="h3").text)

for row in table_data.findAll(name="tr")[1:]:
    product=row.findAll(name="td")[0].text
    production=int(row.findAll(name="td")[1].text)
    export=int(row.findAll(name="td")[2].text)
    print product, production*1000, export*1000