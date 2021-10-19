# Import packages
import json
from datetime import datetime
import math

import pandas as pd
import numpy as np



# Uniswap v3 Relation mapping between feeAmount and tickSpacing
feeMapping = {
    "500":10,
    "3000":60,
    "10000":200
}



def tickcalc(input_data):
    return math.floor(
        np.log(math.sqrt(input_data)) / np.log(math.sqrt(1.0001))
    )

def pricetickcalc(input_data):
    return np.log(math.sqrt(input_data))/np.log(math.sqrt(1.0001))

def priceFromTick(input_data):
    return pow(1.0001, input_data)

def DateTimeConverter(input_date):
    return datetime.utcfromtimestamp(input_date).strftime('%Y-%m-%d %H:%M:%S')




class BollingerDayStrategy:
    '''
        Bollinger Band position recommendation and rebalancing
    '''

    def __init__(self, tickSpacing):

        # These are just base variables created for functionality of Bollinger, these will vary drastically for your own startegy

        # Base parameters, as Bollinger bands work on moving average and standard deviation
        self.mavg = 0
        self.std0 = 0
        self.price0 = 0
        self.currentTick = 0

        # Bollinger multipliers, we choose different for low, medium and high risk
        self.LOW_BOLLINGER = 6
        self.MED_BOLLINGER = 4
        self.HIGH_BOLLINGER = 2
        self.PAST_WINDOW = 10


        # Creating dictionaries for response
        self.activeLiquidityPeriod = {"low":0, "medium":0, "high":0}
        self.inactiveDays = {"low":0, "medium":0, "high":0}
        self.numberOfTransactions = {"low":0, "medium":0, "high":0}
        self.totalDays = 0
        self.tickSpacing = tickSpacing
        self.counterDays = 0
        self.date = None

        self.liquidityData = {
        "low": [],
        "medium": [],
        "high": []
        }

        self.currentPosition = {
        "low": {"positionPriceLower": 0, "positionPriceUpper":0},
        "medium": {"positionPriceLower": 0, "positionPriceUpper":0},
        "high": {"positionPriceLower": 0, "positionPriceUpper":0}
        }

        self.lastTradingDay = {
            "low":-10,
            "medium": -10,
            "high": -10
        }


        # Final Output
        self.recommendedPositions = {

            "low":{
                "date":[],
                "positionPriceLower": [],
                "positionPriceUpper": [],
                },
            "medium":{
                "date":[],
                "positionPriceLower": [],
                "positionPriceUpper": [],
                },
            "high":{
                "date":[],
                "positionPriceLower": [],
                "positionPriceUpper": [],
                }


        }



    def needsRebalancing(self, DAY_COUNTER):
        # If Position goes out of range, rebalance

        res = []
        if(float(self.price0[DAY_COUNTER])>=float(self.currentPosition["low"]["positionPriceLower"]) and float(self.price0[DAY_COUNTER])<=float(self.currentPosition["low"]["positionPriceUpper"])):
            res.append(False)
        else:
            res.append(True)

        if(float(self.price0[DAY_COUNTER])>=float(self.currentPosition["medium"]["positionPriceLower"]) and float(self.price0[DAY_COUNTER])<=float(self.currentPosition["medium"]["positionPriceUpper"])):
            res.append(False)
        else:
            res.append(True)

        if(float(self.price0[DAY_COUNTER])>=float(self.currentPosition["high"]["positionPriceLower"]) and float(self.price0[DAY_COUNTER])<=float(self.currentPosition["high"]["positionPriceUpper"])):
            res.append(False)
        else:
            res.append(True)

        return res


    # Creating position recommendation and rebalancer within the same one using Bollinger Bands
    def positionRecommender(self, DAY_COUNTER):

        positionResponse = {
            "liquidityData":
            {
                "low":[],
                "medium":[],
                "high":[]
            }
        }

        # Getting current tick from price
        currentTick = tickcalc(self.price0[DAY_COUNTER])
        currentTick_ = round(pricetickcalc(self.price0[DAY_COUNTER])/self.tickSpacing)*self.tickSpacing

        import pdb; pdb.set_trace()
        lool = 1

        # Low-risk strategy, standard Bollinger formula,  val - std*BOLLINGER_WIDTH
        llow = self.price0[DAY_COUNTER] - self.std0[DAY_COUNTER] * self.LOW_BOLLINGER
        lhigh = self.price0[DAY_COUNTER] + self.std0[DAY_COUNTER] * self.LOW_BOLLINGER

        # Edge case for negative price on band
        llow = self.price0[DAY_COUNTER] - (self.price0[DAY_COUNTER]/32) if llow<=0 else llow 
        
        import pdb; pdb.set_trace()
        l_lowerTick = round(pricetickcalc(llow)/self.tickSpacing)*self.tickSpacing
        l_higherTick = round(pricetickcalc(lhigh)/self.tickSpacing)*self.tickSpacing




        # Edge case handling for same ticks, usually happens for USDC/USDT pools, can be modified according to your own STRATEGY
        if (l_lowerTick == l_higherTick):
            llow = priceFromTick(math.floor(currentTick/self.tickSpacing) * self.tickSpacing)
            lhigh = priceFromTick(math.ceil(currentTick/self.tickSpacing) * self.tickSpacing)


        # Add the recommendation to the response dictionary
        positionResponse["liquidityData"]["low"].append(( str(max(llow, 0)), str(max(self.price0[DAY_COUNTER], 0)), str(max(lhigh, 0))))

        # Medium-risk strategy
        mlow = self.price0[DAY_COUNTER] - self.std0[DAY_COUNTER] * self.MED_BOLLINGER
        # mhigh = self.price0[DAY_COUNTER] * self.price0[DAY_COUNTER]/mlow
        mhigh = self.price0[DAY_COUNTER] + self.std0[DAY_COUNTER] * self.MED_BOLLINGER
        mlow = self.price0[DAY_COUNTER] - (self.price0[DAY_COUNTER]/32) if mlow<=0 else mlow 
        

        m_lowerTick = round(pricetickcalc(mlow)/self.tickSpacing)*self.tickSpacing
        m_higherTick = round(pricetickcalc(mhigh)/self.tickSpacing)*self.tickSpacing

        # Edge case handling
        if (m_lowerTick == m_higherTick):
            mlow = priceFromTick(math.floor(currentTick/self.tickSpacing) * self.tickSpacing)
            mhigh = priceFromTick(math.ceil(currentTick/self.tickSpacing) * self.tickSpacing)


        # Add the recommendation to the response dictionary
        positionResponse["liquidityData"]["medium"].append(( str(max(mlow, 0)), str(max(self.price0[DAY_COUNTER], 0)), str(max(mhigh, 0))))

        # High-risk strategy
        hlow = self.price0[DAY_COUNTER] - self.std0[DAY_COUNTER] * self.HIGH_BOLLINGER
        hhigh = self.price0[DAY_COUNTER] + self.std0[DAY_COUNTER] * self.HIGH_BOLLINGER
        # hhigh = self.price0[DAY_COUNTER] * self.price0[DAY_COUNTER]/hlow
        hlow = self.price0[DAY_COUNTER] - (self.price0[DAY_COUNTER]/32) if hlow<=0 else hlow 

        h_lowerTick = round(pricetickcalc(hlow)/self.tickSpacing)*self.tickSpacing
        h_higherTick = round(pricetickcalc(hhigh)/self.tickSpacing)*self.tickSpacing

        # Edge case handling
        if (h_lowerTick == h_higherTick):
            hlow = priceFromTick(math.floor(currentTick/self.tickSpacing) * self.tickSpacing)
            hhigh = priceFromTick(math.ceil(currentTick/self.tickSpacing) * self.tickSpacing)

        # Add the recommendation to the response dictionary
        positionResponse["liquidityData"]["high"].append(( str(max(hlow, 0)), str(max(self.price0[DAY_COUNTER], 0)), str(max(hhigh, 0))))


        return positionResponse

    def simulate(self):
        # Running the strategy over the data provided

        # We need to run it based on PAST_WINDOW as for those initial days bollinger gives NaN, because it needs those days atleast
        # to calculate it's moving average and standard deviation, hence we start from PAST_WINDOW-1 counter
        for i in range(self.PAST_WINDOW-1, self.totalDays-1):
            self.counterDays += 1

            # Check if rebalance is needed for low, medium and high strategies
            rebalance = self.needsRebalancing(i+1)
            if(rebalance[0] or rebalance[1] or rebalance[2]):
                pos = self.positionRecommender(i+1)


                if(rebalance[0]): # Fow low risk
                    self.inactiveDays["low"] += 1

                    # Rebalance
                    # print("Rebalance 0 called")
                    
                    self.lastTradingDay["low"] = self.counterDays
                    self.currentPosition["low"]["positionPriceLower"] = pos["liquidityData"]["low"][0][0]
                    self.currentPosition["low"]["positionPriceUpper"] = pos["liquidityData"]["low"][0][2]
                    self.recommendedPositions["low"]["positionPriceLower"].append(self.currentPosition["low"]["positionPriceLower"])
                    self.recommendedPositions["low"]["positionPriceUpper"].append(self.currentPosition["low"]["positionPriceUpper"])
                    self.recommendedPositions["low"]["date"].append(self.date[i])
                    self.numberOfTransactions["low"] += 1
                    


                if(rebalance[1]): # Fow medium risk
                    self.inactiveDays["medium"] += 1

            
                    self.lastTradingDay["medium"] = self.counterDays
                    self.currentPosition["medium"]["positionPriceLower"] = pos["liquidityData"]["medium"][0][0]
                    self.currentPosition["medium"]["positionPriceUpper"] = pos["liquidityData"]["medium"][0][2]
                    self.recommendedPositions["medium"]["positionPriceLower"].append(self.currentPosition["medium"]["positionPriceLower"])
                    self.recommendedPositions["medium"]["positionPriceUpper"].append(self.currentPosition["medium"]["positionPriceUpper"])
                    self.recommendedPositions["medium"]["date"].append(self.date[i])
                    self.numberOfTransactions["medium"] += 1

                if(rebalance[2]): # For high risk
                    self.inactiveDays["high"] += 1

                  
                    self.lastTradingDay["high"] = self.counterDays
                    self.currentPosition["high"]["positionPriceLower"] = pos["liquidityData"]["high"][0][0]
                    self.currentPosition["high"]["positionPriceUpper"] = pos["liquidityData"]["high"][0][2]
                    self.recommendedPositions["high"]["positionPriceLower"].append(self.currentPosition["high"]["positionPriceLower"])
                    self.recommendedPositions["high"]["positionPriceUpper"].append(self.currentPosition["high"]["positionPriceUpper"])
                    self.recommendedPositions["high"]["date"].append(self.date[i])
                    self.numberOfTransactions["high"] += 1
    def setValues(self, transactionResponse):
        
        # Calculating parameter values values
        self.mavg = pd.Series(transactionResponse["graphData"]["token1Price"]).rolling(self.PAST_WINDOW).mean()
        self.std0 = pd.Series(transactionResponse["graphData"]["token1Price"]).rolling(self.PAST_WINDOW).std()
        self.price0 = transactionResponse["graphData"]["token1Price"]
        self.price0[0] = self.price0[1]
        self.totalDays = len(self.price0)
        self.date = transactionResponse["graphData"]["datetime"]



def lambda_handler(event, context):

    # Creating output placeholders
    response = {}
    transactionResponse = {}

    try:
        # Getting the payload
        event = json.loads(event)
        payload = event["body"]

        feeAmount = payload["feeAmount"]
        # This is useful for determing the tick positions
        tickSpacing = feeMapping[str(feeAmount)]



    except Exception as e:

        # If any error faced in parsing the above keys

        s = str(e)
        print(s)
        response["statusCode"] = 400
        transactionResponse["error"] = True
        transactionResponse["message"] = s 
        # transactionResponse["file"] = ""
        # transactionResponse["bucket"] = OUTPUT_BUCKET
        response["headers"] = {}
        response["headers"]["Content-Type"] = "application/json"
        # transactionResponse["score"] = "{:.5f}".format(0.00000)

        response["body"] = transactionResponse

        # Converting python dictionary to JSON Object
        response_JSON = response
        
        # Revert the Response
        return response_JSON


    try:
        # Reading the training file
        responseDf = pd.read_csv("competition_data/sample.csv")


        # Now we want only those rows where the price is actually changing, THIS CAN VARY FOR YOUR STRATEGY, THIS IS JUST FOR EXAMPLE
        newDf = responseDf["token1Price"].diff()
        responseDf = responseDf[newDf != 0.0]
        responseDf = responseDf.reset_index(drop=True)


       
        # Getting the data from the csv for strategy building, this will be passed to bollinger strategy later
        transactionResponse["graphData"] = {
            "sqrtPrice":list(responseDf["sqrtPrice"].astype(str)),
            "tick": list(responseDf["tick"].astype(str)),
            "price":list(pow(1.0001, responseDf["tick"].astype(float))),
            "token0Price": list(pd.to_numeric(responseDf["token0Price"])),
            "token1Price": list(pd.to_numeric(responseDf["token1Price"])),
            "datetime": responseDf["datetime"]
        }


        # Strategy Section - Bollinger
        # Creating the object for strategy
        bollinger1 = BollingerDayStrategy(tickSpacing)
        print("Object Created")

        # Passing the data
        bollinger1.setValues(transactionResponse)
        print("Values set")

        # Running the strategy
        bollinger1.simulate()

        # Setting output values
        transactionResponse["bollingerPositionRecommendations"] = bollinger1.recommendedPositions

        # Checking the output
        print("Bollinger response test")
        print(transactionResponse["bollingerPositionRecommendations"])






    except Exception as e:
        # If any error faced in parsing the above keys
        s = str(e)
        print(e)

        response["statusCode"] = 400
        transactionResponse["error"] = True
        transactionResponse["message"] = s 
        # transactionResponse["file"] = ""
        # transactionResponse["bucket"] = OUTPUT_BUCKET
        response["headers"] = {}
        response["headers"]["Content-Type"] = "application/json"
        # transactionResponse["score"] = "{:.5f}".format(0.00000)

        response["body"] = transactionResponse

        # Converting python dictionary to JSON Object
        response_JSON = response
        
        # Revert the Response
        return response_JSON




    response["statusCode"] = 200
    transactionResponse["error"] = False
    transactionResponse["message"] = "Error free execution"
    # transactionResponse["file"] = output_path
    # transactionResponse["bucket"] = OUTPUT_BUCKET
    response["headers"] = {}
    response["headers"]["Content-Type"] = "application/json"
    # transactionResponse["score"] = "{:.5f}".format(0.00000)

    response["body"] = transactionResponse

    # Converting python dictionary to JSON Object
    response_JSON = response

    return response





if __name__ == "__main__":


    # This is just to make the data JSON style called, nothing particular, also it makes it lambda-compatible if needed
    input_data = {
          "body": {
            "address": "0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8",
            "feeAmount":3000 # This is a required variable
          }
        }

    # Calling the handler
    input_json = json.dumps(input_data)
    print("Test JSON")
    print(input_json)
    result = lambda_handler(input_json,"Context")
    # print("Lambda Result")
    # print(result)













































