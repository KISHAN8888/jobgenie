import os

import docx
import google.generativeai as genai
from langchain.agents import tool  
import pandas as pd

genai.configure(api_key="AIzaSyBgXXTa043ZpKYxIYfGOQ6P91fBD6Bkfxk")

model = genai.GenerativeModel('gemini-1.5-flash')

class JobSearchToolset():
    
    @tool
    def search_jobs(self, query: str):
        """Search for job listings based on the query."""
        # Implement the job search logic here
        return SomeTool.search(query)
    
    @tool
    def extract_text_from_docx(self, file_path: str):
        doc = docx.Document(file_path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)
  
    @tool
    def get_skills_and_roles(self, resume_text: str):
        # Load the pre-trained language model pipeline for text generation
        skillstext = "This is a resume. Give all the skills of the candidate separated with commas." + resume_text
        rolestext = "This is a resume, follow this format strictly. format: Give all the role names only to which candidate can apply separated with commas." + resume_text
        skills = model.generate_content(skillstext).text
        roles = model.generate_content(rolestext).text
        return skills, roles
  
    @staticmethod
    def get_skill_names(skill_match_statuses):
        return [skill['skill']['name'] for skill in skill_match_statuses]
  
    @staticmethod
    def extract_job_info(api, jobs):
        job_info = []
        for job in jobs:
            job_title = job.get('title')
            tracking_urn = job.get('trackingUrn')
            
            if tracking_urn:
                job_id = tracking_urn.split(':')[-1]
                job_skills_data = api.get_job_skills(job_id)
                skill_match_statuses = job_skills_data.get('skillMatchStatuses', [])
                job_skills = JobSearchToolset.get_skill_names(skill_match_statuses)
                job_link = f"https://www.linkedin.com/jobs/view/{job_id}/"
                job_info.append({'urnlijobposting': tracking_urn, 'link': job_link, 'title': job_title, 'skills': job_skills})
        
        return job_info
    
    @tool
    def search_jobs_by_keywords_location(self, api, keywords: str, location_name: str):
        jobs = api.search_jobs(keywords=keywords, location_name=location_name)
        job_info = self.extract_job_info(api, jobs)
        df = pd.DataFrame(job_info)
        return df
    
    @tool
    def find_similar_jobs(self, url: str):
        """Find similar job listings based on a given URL."""
        # Implement logic to find similar job listings
        return SomeTool.find_similar(url)
    
    @tool
    def get_job_details(self, ids: str):
        """Get details of a job listing based on provided IDs."""
        ids = eval(ids)
        # Implement logic to get job details
        return SomeTool.get_details(ids)
    
    
    
def tools():
    return [
      JobSearchToolset.search_jobs,
      JobSearchToolset.find_similar_jobs,
      JobSearchToolset.get_job_details
    ]