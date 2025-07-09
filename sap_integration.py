import requests

def get_ticket_status(ticket_id, access_token):
    url = f"https://sap.api.com/tickets/{ticket_id}"
    headers = {
        "Authorization": f"Bearer {access_token}"
    }
    response = requests.get(url, headers=headers)
    return response.json()