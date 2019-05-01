import requests

url ='https://api.data.gov/ed/collegescorecard/v1/schools?school.name=boston%20college&api_key=EGoNSZQGOhEejyiFV4J991cPhlcQ9bqeCJy3y8OD'

result=requests.get(url)

print('Status:', result.status_code)
print('Header: ', result.headers)
print('Encoding: ', result.encoding)
print('Text:', result.text)
print('JSON:', result.json())