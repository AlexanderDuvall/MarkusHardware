import requests
import json

# some JSON:
x = '{ "id":1}'
# parse x:
y = json.loads(x)
if __name__ == '__main__':
    url = "http://localhost:9696/test"
    r = requests.get(url, json=y)
    data = r.json()
    print(data["data"])
