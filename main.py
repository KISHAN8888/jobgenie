from dotenv import load_dotenv
from crewai import Crew
from tasks import JobApplicationTasks
from agents import JobApplicationAgents
from dotenv import load_dotenv
import os
load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI





def main():

    ## call the gemini models
    llm= ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                           verbose=True,
                           temperature=0.5,
                           google_api_key=os.getenv("GOOGLE_API_KEY"))



    load_dotenv()
    
    print("## Welcome to the Job Application Assistant")
    print('-------------------------------')
    user_data = input("Please provide your data and preferences (comma-separated):\n")
    resume = input("Please upload your resume:\n")
    
    tasks = JobApplicationTasks()
    agents = JobApplicationAgents()
    
    # create agents
    profile_analysis_agent = agents.profile_analysis_agent()
    job_search_agent = agents.job_search_agent()
    recommendation_agent = agents.recommendation_agent()
    feedback_agent = agents.feedback_agent()
    
    # create tasks
    profile_analysis_task = tasks.profile_analysis_task(profile_analysis_agent, resume)
    job_search_task = tasks.job_search_task(job_search_agent, profile_analysis_task)
    recommendation_task = tasks.recommendation_task(recommendation_agent, profile_analysis_task, job_search_task)
    feedback_task = tasks.feedback_task(feedback_agent, recommendation_task, user_data)
    
    crew = Crew(
      agents=[
        profile_analysis_agent,
        job_search_agent,
        recommendation_agent,
        feedback_agent
      ],
      tasks=[
        profile_analysis_task,
        job_search_task,
        recommendation_task,
        feedback_task
      ]
    )
    
    result = crew.kickoff()
    
    print(result)
    
if __name__ == "_main_":
    main()