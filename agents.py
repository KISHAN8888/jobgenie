from textwrap import dedent
from crewai import Agent

class JobApplicationAgents():
    def profile_analysis_agent(self):
      return Agent(
        role="Profile Analysis Specialist",
        goal='Analyze user resume to identify key skills and potential roles',
        backstory=dedent("""\
          As a Profile Analysis Specialist, your mission is to scrutinize the user's resume,
          extracting key skills and identifying roles that match their profile."""),
        verbose=True
      )
      
    def job_search_agent(self):
      return Agent(
        role='Job Search Specialist',
        goal='Find jobs matching the identified skills and roles',
        backstory=dedent("""\
            As a Job Search Specialist, your role is to search for job listings
            that match the identified skills and roles from the user's profile."""),
        verbose=True,
        tools=[tool],
        llm=llm,
        allow_delegation=True
      )
      
    def recommendation_agent(self):
      return Agent(
        role='Recommendation Advisor',
        goal='Recommend top job matches based on user skills',
        backstory=dedent("""\
            As a Recommendation Advisor, your task is to recommend the top job
            matches based on the user’s skills and the jobs found."""),
        verbose=True,
        verbose=True,
        tools=[tool],
        llm=llm,
        allow_delegation=True
      )
      
    def feedback_agent(self): 
      return Agent(
        role='Feedback Coordinator',
        goal='Provide feedback on job matches and suggested improvements',
        backstory=dedent("""\
            As the Feedback Coordinator, your role is to compile recommendations
            and provide feedback to the user on job matches and necessary improvements."""),
        verbose=True,
        verbose=True,
        tools=[tool],
        llm=llm,
        allow_delegation=True
      )