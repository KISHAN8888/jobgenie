from linkedin_api import Linkedin
import browser_cookie3
import requests
import pandas as pd

class LinkedInJobSearch:
    def __init__(self, username, password):
        self.api = self._authenticate(username, password)

    def _authenticate(self, username, password):
        cookiejar_simple = browser_cookie3.firefox(domain_name='.linkedin.com')
        cookiejar = requests.cookies.RequestsCookieJar()
        for cookie in cookiejar_simple:
            cookiejar.set_cookie(cookie)
        
        api = Linkedin(username, password, cookies=cookiejar)
        return api

    def get_skill_names(self, skill_match_statuses):
        return [skill['skill']['name'] for skill in skill_match_statuses]

    def extract_job_info(self, jobs):
        job_info = []
        for job in jobs:
            job_title = job.get('title')
            tracking_urn = job.get('trackingUrn')

            if tracking_urn:
                job_id = tracking_urn.split(':')[-1]
                job_skills_data = self.api.get_job_skills(job_id)
                skill_match_statuses = job_skills_data.get('skillMatchStatuses', [])
                job_skills = self.get_skill_names(skill_match_statuses)
                job_link = f"https://www.linkedin.com/jobs/view/{job_id}/"
                job_info.append({'urnlijobposting': tracking_urn, 'link': job_link, 'title': job_title, 'skills': job_skills})
        
        return job_info

    def search_jobs(self, keywords, location_name):
        jobs = self.api.search_jobs(keywords=keywords, location_name=location_name)
        job_info = self.extract_job_info(jobs)
        df = pd.DataFrame(job_info)
        return df

# Usage example
username = 'your-email@example.com'
password = 'your-password'

# Initialize the LinkedInJobSearch object once
linkedin_search = LinkedInJobSearch(username, password)

# Search for jobs and get the DataFrame
keywords = "system engineer"
location_name = "Berlin"
df = linkedin_search.search_jobs(keywords, location_name)

# Print the DataFrame
print(df)
