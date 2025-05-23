import requests
from typing import Optional, Dict, Any, Union

class DiscordWebhook:
    def __init__(self, webhook_url: str):
        """
        Initialize the DiscordWebhook class with a webhook URL.
        
        Args:
            webhook_url (str): The Discord webhook URL to send messages to
        """
        self.webhook_url = webhook_url
        
    def send_message(
        self,
        content: Optional[str] = None,
        username: Optional[str] = None,
        avatar_url: Optional[str] = None,
        embeds: Optional[list] = None,
        **kwargs
    ) -> requests.Response:
        """
        Send a message to Discord using the webhook.
        
        Args:
            content (str, optional): The message content
            username (str, optional): Override the default username
            avatar_url (str, optional): Override the default avatar
            embeds (list, optional): List of embed objects
            **kwargs: Additional parameters to include in the payload
            
        Returns:
            requests.Response: The response from the Discord API
        """
        payload: Dict[str, Any] = {}
        
        if content:
            payload["content"] = content
        if username:
            payload["username"] = username
        if avatar_url:
            payload["avatar_url"] = avatar_url
        if embeds:
            payload["embeds"] = embeds
            
        # Add any additional parameters
        payload.update(kwargs)
        
        response = requests.post(
            self.webhook_url,
            json=payload
        )
        
        return response

# Example usage:
if __name__ == "__main__":
    # Replace with your webhook URL
    webhook_url = "YOUR_WEBHOOK_URL_HERE"
    
    # Create webhook instance
    webhook = DiscordWebhook(webhook_url)
    
    # Send a simple message
    response = webhook.send_message(
        content="Hello from Python!",
        username="Custom Bot Name",
        avatar_url="https://example.com/avatar.png"
    )
    
    # Check if the message was sent successfully
    if response.status_code == 204:
        print("Message sent successfully!")
    else:
        print(f"Failed to send message. Status code: {response.status_code}")
        print(f"Response: {response.text}") 