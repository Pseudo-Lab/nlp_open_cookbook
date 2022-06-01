import requests

headers = {'Content-Type': 'application/json',}
ip = "14.49.44.212"
port = "1000"
json_data = {"text":["영화 재미있다 ㅎㅎ", "재미 진짜 없다"]}

response = requests.post('http://{}:{}/predict'.format(ip, port), headers=headers, json=json_data)
print(response.text)