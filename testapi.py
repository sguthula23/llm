# import requests
# response = requests.get('http://127.0.0.1:5000/status')
# print(response.text)


import requests
import json

# Define the URL of the API endpoint
url = "http://localhost:8080/predict"

# Define the data you want to send to the API
# data = {
#     'input': 'Which college Sswathi studied'
# }
# Convert the data to JSON format
# json_data = json.dumps(data)
# print("JSON Data", json_data)
# Send a POST request to the API
response = requests.get(url)
# Print the response from the API
print(response.text)


url = "http://localhost:8080/predictquery"
# Define the data you want to send to the API
data = {
    'input': 'period of performance for CISA'
}
# Convert the data to JSON format
# json_data = json.dumps(data)
# print("JSON Data", json_data)
# Send a POST request to the API
print(data['input'])
response = requests.post(url, json=data['input'])
# Print the response from the API
print(response.text)

