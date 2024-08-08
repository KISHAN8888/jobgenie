from textwrap import dedent
from crewai import Task

class JobApplicationTasks():
  def profile_analysis_task(self, agent, resume):
    return Task(
      description=dedent(f"""\
        Analyze the provided resume to extract key skills and identify roles
        that the user is suitable for based on their experience and skills.

        Resume: {resume}"""),
      expected_output=dedent("""\
        A detailed report summarizing the key skills extracted from the resume
        and a list of potential roles."""),
      agent=agent,
      async_execution=True
    )
    
  def job_search_task(self, agent, profile_analysis_task):
    return Task(
			description=dedent(f"""\
				Search for jobs that match the skills and roles identified from the
				profile analysis. Use job search engines or databases to find relevant listings.

				Profile Analysis: {profile_analysis_task}"""),
			expected_output=dedent("""\
				A list of job listings that match the identified skills and roles, including links."""),
			async_execution=True,
			agent=agent
		)
    
  def recommendation_task(self, agent, profile_analysis_task, job_search_task):
    return Task(
			description=dedent(f"""\
				Recommend the top job matches for the user based on the skills identified
				in the profile analysis and the jobs found in the job search task.

				Profile Analysis: {profile_analysis_task}
				Job Search Results: {job_search_task}"""),
			expected_output=dedent("""\
				A prioritized list of top job matches, with recommendations on which jobs to apply for."""),
			agent=agent
		)
    
  def feedback_task(self, agent, recommendation_task, user_data):
    return Task(
			description=dedent(f"""\
				Provide feedback to the user on the top job matches and suggest any
				improvements or additional skills that could enhance their job applications.

				Recommendations: {recommendation_task}
				User Data: {user_data}"""),
			expected_output=dedent("""\
				A feedback report that includes the top job matches, suggestions for improvements,
				and any additional skills or actions needed to increase the chances of success."""),
			agent=agent
		)