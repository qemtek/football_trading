FPL Betting on Betfair Exchange (pre-match and in-play) #2

Link to trello:
https://trello.com/b/8okRfP4A/football-prediction-betfair-exchange

Stage 1
- Get poisson distribution mean/sd for every EPL game in the last 10 seasons
- Use this data to get probabilities of each correct score in the betfair market (up to 3 goals for each team then any other win for each team)
- Check for variation throughout the years to see if we can use older data.

Stage 2
- Gather correct store odds (pre-game) for all games
- Use the calculated probabilities and betting odds to compare different betting strategies and measure total return. [betting strategies: standard betting, kelly staking]
- Test different window sizes (how many games should we use as history)

Stage 3
- Use additional features to change the probabilities (by altering the mean of the poisson distribution). Here we basically turn the mean of the poisson distribution into the output of a model, using other features.
...

Stage x
- Incorporate in-play features. Cash out and back in reaction to events happening in the game using a recreational neural network/reinforcement learning or similar!



LOG 

15-06-2019 - Tried to get horse racing data out of britishhorseracing.com but ran indoor difficulties decoding the response, it seems to be in HTML when it should be JSON? To look into horse racing data about the ground and horses (time form or similar) is needed for good results. Cant access the football data for a week or so because of errors with bet fair historic data.

14-06-2019 - Rolling mean data for every FPL game since 2003 has been added to the database, along with poisson probabilities for goals for and goals against for each team. Also added score probabilities from this, need to work on the logic for pulling all 

09-06-2019 - Managed to parse historic data, but the package I am using is based around horse racing , so there are two options. Edit the code to include football data, or use a different package that also can use historical data (I think historic and non historic are in the same format).

10-06-2019 - Found out how to get all the data I need, now I need to turn the historical data into a function and call it on every
