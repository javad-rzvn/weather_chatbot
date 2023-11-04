import streamlit as st
import os
import requests
import matplotlib.pyplot as plt
from geopy import geocoders
from dotenv import load_dotenv
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.chat_models import ChatOpenAI
from streamlit_chat import message
import io
import tempfile
from langchain.schema import OutputParserException



def get_weather_data(lat: float, lon: float):
    """
      This function gets lat and lon then fetch weather data

      :param float lat: latitude of the point
      :param float lon: longitude of the point
    """

    base_url = "https://archive-api.open-meteo.com/v1/archive?start_date=2022-01-01&end_date=2022-12-31&daily=temperature_2m_mean&timezone=GMT"
    params = {
        "latitude": lat,
        "longitude": lon,
    }

    try:
        response = requests.get(base_url, params=params)
        weather_data = response.json()

        time_values = weather_data["daily"]["time"]
        temperature_2m_values = weather_data["daily"]["temperature_2m_mean"]

        plt.rc('font', size=12)
        fig, ax = plt.subplots(figsize=(10, 6))

        # Specify how our lines should look
        ax.plot(time_values, temperature_2m_values, color='tab:orange', label='Temperature')
        plt.xticks(rotation=90)
        ax.set_xticks(ax.get_xticks()[::10])

        # Same as above
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature °C')
        ax.set_title(f'Lat: {lat}, Lon: {lon}')
        ax.grid(True)
        ax.legend(loc='upper left')

        # Get the (plotted) image into memory file
        # imgdata = BytesIO()
        # fig.savefig(imgdata, dpi=60, format='png')
        # imgdata.seek(0)  #rewind the data
        # imgJpg = imgdata.getvalue()

        # html = """<html><body><img src="data:image/png;base64,{}"/></body></html>""".format(base64.encodebytes(imgdata.getvalue()).decode())

        plt_name = "plot_" + str(lat).split(".")[0] + "_" + str(lat).split(".")[1] + "_" + str(lon).split(".")[0] + "_" + str(lon).split(".")[1]
        plt.savefig(plt_name + ".jpg")

        # buf = io.BytesIO()
        # plt.savefig(buf, format="jpg")
        # #print(buf.getvalue()) return bytes of plot 
        # fp = tempfile.NamedTemporaryFile() 
        # # print(fp.name) return file name 
        # with open(f"{plt_name}.jpg",'wb') as ff:
        #     ff.write(buf.getvalue()) 
        # buf.close()

        # st.pyplot(fig)
        # st.write(fig)
        st.image(plt_name + ".jpg")

        # pickle.dump(fig, open(plt_name + '.pickle', 'wb'))
        print(f"Weather in {lat}, {lon}: ", sum(temperature_2m_values)/len(temperature_2m_values))
        return lat, lon, sum(temperature_2m_values)/len(temperature_2m_values)

    except requests.exceptions.RequestException as e:
        print(f"Error occurred: {e}")
        return None


def parse_lat_lon(string):
    lat, lon = string.split(",")
    return get_weather_data(float(lat), float(lon))

# weather_data = parse_lat_lon("52.52,13.41")
# print(f"Weather in {weather_data[0]}, {weather_data[1]}: {weather_data[2]}")

def get_coordinates(city: str):
  gn = geocoders.GeoNames(username="sinbb")
  output = gn.geocode(city)

  print(f"City: {output[0]}, Location: {output[1]}")
  return str(output[1][0]) + "," + str(output[1][1])

# get_coordinates("Tehran")


def get_router_agent(openai_api_key):
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-0613", temperature=0)

    coordinates_tool = Tool.from_function(
        func=get_coordinates,
        name="FetchCityCoordinates",
        description="useful for when you need to get city coordinates. The input to this tool should be a string, representing the name of the city. For example, `Tehran` would be the input if you wanted to get coordinates for Tehran"
    )

    weather_tool = Tool.from_function(
        func=parse_lat_lon,
        name="WeatherFetcher",
        description="The input to this tool should be a comma separated string of two numbers representing the lat and lon"
    )

    agent_get_coordinates = initialize_agent(
        tools=[coordinates_tool],
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )

    PREFIX = """Answer the following questions as best you can. You have access to the following tools:"""
    FORMAT_INSTRUCTIONS = """Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat 1 times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question
    """

    SUFFIX = """
    the input will be the latitude and longitude of a location and your task is use the apporporaite tool and give back the weather information.
    you may see it is unavailable, if it is, then just return the results directly without saying anything.
    double check for any inaccuracies.



    Begin!
    Question: {input}
    Thought:{agent_scratchpad}"""

    agent_get_weather_data = initialize_agent(
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        tools=[weather_tool],
        llm=llm,
        verbose = False,
        agent_kwargs={
            'prefix':PREFIX,
            'format_instructions':FORMAT_INSTRUCTIONS,
            'suffix':SUFFIX
        },
        return_direct= True,
        handle_parsing_errors=True,
    )

    router_tools = [
        Tool(
        name='Fetch_City_Coordinates',
        func=agent_get_coordinates.run,
        description='First get latitude and longitude of the given city and return it as comma separated string of two numbers. this is always the first tool you need to use'),
        Tool(
        name='Weather_Fetcher',
        func=agent_get_weather_data.run,
        description='useful for when you need to get weather. tell this agent what info you want to get for that specific latitude and longitude.'),
    ]

    router_agent = initialize_agent(
        tools=router_tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=False,
        handle_parsing_errors=True,
    )

    return router_agent


def handel_userinput(openai_api_key, user_question):

    warning = "if unavailable just say '💀', do no say it is unavailable"

    try:
        response = get_router_agent(openai_api_key).run(f"{user_question} {warning}")
    except Exception as e:
        print("error:::", e)
        response = str(e)
        if response.startswith("Parsing LLM output produced both a final answer and a parse-able action"):
            response = response.replace("Parsing LLM output produced both a final answer and a parse-able action", "").replace("`", "")
            return response
        else:
            raise Exception(str(e))
          
    # try:
    st.session_state.chat_history = response

    response_container = st.container()

    # with response_container:
    #     for i, messages in enumerate(st.session_state.chat_history):
    #         if i % 2 == 0:
    #             message(messages.content, is_user=True, key=str(i))
    #         else:
    #             message(messages.content, key=str(i))

    # except OutputParserException as e:
    #     st.session_state.chat_history = "-"
    #     response_container = st.container()


def main():
    # load_dotenv()
    st.set_page_config(page_title="بات وضعیت آب و هوا")
    st.header("بات وضعیت آب و هوا")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "processComplete" not in st.session_state:
        st.session_state.processComplete = None

    with st.sidebar:
        openai_api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")
        # user_prompt = st.text_input("پیغام موردنظر خود را وارد کنید", key="city_name", type="default")
        process = st.button("شروع")
    if process:
        if not openai_api_key:
            st.info("Please add your OpenAI API key to continue.")
            st.stop()

        st.session_state.conversation = get_router_agent(openai_api_key)
        st.session_state.processComplete = True

    
    user_prompt = st.chat_input("پیغام موردنظر خود را وارد کنید")
    if not openai_api_key:
        st.info("Please add your OpenAI API key to continue.")
        st.stop()
    else:
        if user_prompt:
            handel_userinput(openai_api_key, user_prompt)


if __name__ == '__main__':
    main()