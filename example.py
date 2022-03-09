from app.prediction import PersonalityPrediction

p = PersonalityPrediction()
string = {"text": "Wow, only three weeks to 2022! What will 2032 will be like? Seems so futuristic! Will we be on Mars? ||| thinking of quitting my jobs & becoming an influencer full-time wdyt ||| Laws are on one side, poets on the other ||| Let’s set an age limit after which you can’t run for political office, perhaps a number just below 70 … :)"}
print(p.predict(string))