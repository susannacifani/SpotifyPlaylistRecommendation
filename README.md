# SpotifyPlaylistRecommendation
The purpose of this project is to enhance song recommendations based on an existing Spotify playlist using machine learning


## Indice
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)

## Introduction
This project is useful because it makes your Spotify playlists even better. With machine learning, it fine-tunes song recommendations, so you get music that matches your taste, making your listening experience more enjoyable and expanding your knowledge of songs as well

## Installation
To use this code, you will obviously need a Spotify account. 
Go to the following website, https://developer.spotify.com, and log in with your Spotify credentials. 
Next, open your dashboard and hit the “Create an app” button. Set the Redirect URL to “http://localhost:8000" and choose your application name. Then get your client ID and Secret, you'll need them later.
You will also need to download your Spotify listening history, which can take up to 5 or 30 days depending on which type of history you choose to download. I chose the extended listening history.
Download this repository.

## Usage
I removed my Client ID (cid) and my Client Secret (secret) from the script because it's a reserved information.
To run the code, you need to replace 'insertcid' and 'insertsecret' with your Client ID and Secret, 'susannacifani' with your Spotify username, then you can leave 'http://localhost:8000' (if you set the Redirect URL to “http://localhost:8000" during the application setting), you need to replace the next link with the link of the playlist you want to be analyzed to receive a suggested playlist. Then, I've included the name of the folder containing the listening history data and the exact folder with the listening history. Then, I chose the number of months to consider (starting from the moment of the listening history request, the previous months will be analyzed).

main('insertcid', 'insertsecret', 
        'susannacifani', 'http://localhost:8000', 
        'https://open.spotify.com/playlist/2u11ymH4swrfdVmoELwbRb?si=549c0d6111bb4dc6', 
        'MyData', 'StreamingHistory0', 6)

When you run the code for the first time, a Spotify web page will open, and from there, you need to grant all the permissions.
