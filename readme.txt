FPL Betting #2

Stage 1
- Get poisson distribution mean/sd for every EPL game in the last 10 seasons
- Use this data to get probabilities of each correct score in the betfair market (up to 3 goals for each team then any other win for each team)
- Check for variation throughout the years to see if we can use older data.

Stage 2
- Gather correct store odds (pre-game) for all games
- Use the calculated probabilities and betting odds to compare different betting strategies and measure total return. [betting strategies: standard betting, kelly staking]
- Try test different window sizes and report results

Stage 3
- Use additional data to inform the probabilities, using the previous probabilities as a base.. Many different options for this:
	Option 1 - Model with 4 outputs, which are the mean and standard deviation of the goal distributions.
	Option 2 - Model with 1 output per goal selection in betfair (up to 3-3 then 'any other home/away win')

...

Stage x
- Incorporate in-play features. Cash out and back in reaction to events happening in the game using a recreational neural network or similar!
