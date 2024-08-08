
from linkedin_api import Linkedin
import browser_cookie3
import requests
import docx
import google.generativeai as genai
import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from serpapi import GoogleSearch
import json

import nltk

import pandas as pd
import requests
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import docx
import google.generativeai as genai

import numpy as np
import os


# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)



cookiejar_simple = browser_cookie3.firefox(domain_name='.linkedin.com')
cookiejar = requests.cookies.RequestsCookieJar()
for cookie in cookiejar_simple:
    cookiejar.set_cookie(cookie)   

email = os.getenv('LINKEDIN_EMAIL')
password = os.getenv('LINKEDIN_PASSWORD')
api = Linkedin(email, password, cookies=cookiejar)
api_key = os.getenv('GOOGLE_API_KEY')
API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
API_TOKEN = os.getenv('HUGGINGFACE_API_TOKEN')
headers = {"Authorization": f"Bearer {API_TOKEN}"}





keywords = input("Enter the keywords: ")
location_name = input("Enter the location: ")

# Call the search_jobs function with the specified parameters
jobs = api.search_jobs(keywords=keywords, location_name=location_name)

# Print or process the search results
print(jobs)


len(jobs)


def get_skill_names(skill_match_statuses):
    return [skill['skill']['name'] for skill in skill_match_statuses]


def extract_job_info(jobs):
    job_info = []
    for job in jobs:
        job_title = job.get('title')
        tracking_urn = job.get('trackingUrn')
        
        if tracking_urn:

            
            job_id = tracking_urn.split(':')[-1]
            job_skills_data = api.get_job_skills(job_id)
            skill_match_statuses = job_skills_data.get('skillMatchStatuses', [])
            job_skills = get_skill_names(skill_match_statuses)
            job_link = f"https://www.linkedin.com/jobs/view/{job_id}/"
            job_info.append({'urnlijobposting': tracking_urn, 'job_id': job_id, 'link': job_link,'title':job_title, 'skills':job_skills})
    return job_info



# Extract job information
job_info = extract_job_info(jobs)

print(job_info)
# Print the extracted information
for job in job_info:
    print(job)


job_data = pd.DataFrame(job_info)
print(job_data)




def configure_genai(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel('gemini-1.5-flash')

# Function to extract text from DOCX file
def extract_text_from_docx(file_path):
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

# Function to get skills and suitable roles from LLM
def get_skills_and_roles(model, resume_text):
    skillstext = "This is a resume. Give all the skills of the candidate separated with commas. " + resume_text
    rolestext = "This is a resume, follow this format strictly. Format: Give all the role names only to which candidate can apply separated with commas. " + resume_text
    skills2 = model.generate_content(skillstext).text
    roles = model.generate_content(rolestext).text
    return skills2, roles

# Function to process resume and get skills and roles
def process_resume(api_key, file_path):
    model = configure_genai(api_key)
    
    # Extract text from resume
    resume_text = extract_text_from_docx(file_path)
    print("Resume Text Extracted:\n", resume_text)

    # Get skills and suitable roles using LLM
    skills2, roles = get_skills_and_roles(model, resume_text)
    print("\nExtracted Skills:\n", skills2)
    print("\nSuitable Roles:\n", roles)
    return skills2, roles






 
file_path = "C:\\Users\\kisha\\Documents\\jobease\\Kishan Tripathi Resume.docx"
skills2, roles = process_resume(api_key, file_path)


print(skills2)






def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# User skills
user_skills = [skill.strip() for skill in skills2.split(',')]

# Combine all skills into a single string
user_skills_text = ", ".join(user_skills)

def calculate_job_match(row):
    # Combine title and skills (if available) for job text
    job_text = row['title']
    if isinstance(row['skills'], str) and row['skills'] != '[]':
        job_text += " " + row['skills']
    
    # Calculate similarity
    payload = {
        "inputs": {
            "source_sentence": user_skills_text,
            "sentences": [job_text]
        }
    }
    result = query(payload)
    
    # The API returns a list of similarities, we only sent one job so we take the first (and only) result
    return result[0] if isinstance(result, list) and len(result) > 0 else 0

# Calculate match scores for all jobs
job_data['match_score'] = job_data.apply(calculate_job_match, axis=1)

# Sort jobs by match score in descending order
top_jobs = job_data.sort_values('match_score', ascending=False).head(5)

# Print results
for _, job in top_jobs.iterrows():
    print(f"Title: {job['title']}")
    print(f"Match score: {job['match_score']:.2f}")
    print(f"Link: {job['link']}")
    print(f"Skills: {job['skills']}")
    print(f"Job_id: {job['job_id']}")
    print()


results = []
for _, job in top_jobs.iterrows():
    job_result = {
        "title": job['title'],
        "match_score": job['match_score'],
        "link": job['link'],
        "skills": job['skills'],
        "job_id": job['job_id']
        
    }
    results.append(job_result)

# Return results (you can further process or display these as needed)
print(results)



job_ids = [job['job_id'] for job in results]



print(job_ids)


def get_job(job_ids):
    job_datas = []
    for job_id in job_ids:
        job_data = api.get_job(job_id)
        job_datas.append(job_data)
        print(f"Iteration {len(job_datas)}: job_id={job_id}, job_data={job_data}")

    

    return job_datas


job_datas = get_job(job_ids)


print(job_datas)


def extract_job_details(api_response):
    # Extract company name
    company_name = api_response['companyDetails']['com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany']['companyResolutionResult']['name']
    
    # Extract job title
    title = api_response['title']
    #job url

    



    # Extract job location
    location = api_response['formattedLocation']
    
    # Extract remote work allowed
    remote_work_allowed = api_response['workRemoteAllowed']
    
    # Extract job description
    description = api_response.get('description', {}).get('text')
    
    # Extract job details
    job_details = {
        'job_title': title,
        'company': company_name,
        'location': location,
        'remote_work_allowed': remote_work_allowed,
        'job_description': {
            'description': description
        },
        
    }
    
    # Extract company details
    company_url = api_response['companyDetails']['com.linkedin.voyager.deco.jobs.web.shared.WebCompactJobPostingCompany']['companyResolutionResult']['url']
    company_details = {
        'company_name': company_name,
        'company_url': company_url
    }
    
    # Extract employment type and seniority level
    employment_type = api_response.get('employmentType')
    seniority_level = api_response.get('seniorityLevel')
    
    # Extract job functions and industries
    job_functions = [func['name'] for func in api_response.get('jobFunctions', [])]
    industries = [industry['name'] for industry in api_response.get('industries', [])]
    
    # Extract required skills
    technical_skills = [skill['name'] for skill in api_response.get('technicalSkills', [])]
    other_skills = [skill['name'] for skill in api_response.get('otherSkills', [])]
    soft_skills = [skill['name'] for skill in api_response.get('softSkills', [])]
    
    # Prepare job details object
    job_details.update({
        'job_details': {
            'employment_type': employment_type,
            'seniority_level': seniority_level,
            'job_functions': job_functions,
            'industries': industries,
            'remote_work_allowed': remote_work_allowed,
            'job_description': {
                'description': description
            },
            'job_url': api_response['applyMethod']
        },
        'company_details': {
            'company_name': company_name,
            'company_url': company_url
        },
        'required_experience_and_skills': {
            'technical_skills': technical_skills,
            'other_skills': other_skills,
            'soft_skills': soft_skills
        }
    })
    
    return job_details





def process_job_data(job_data_list):
    results2 = []
    for job_data in job_data_list:
        job_details = extract_job_details(job_data)
        results2.append(job_details)
        print(f"Iteration {len(results2)}: job_details={job_details}")
    print(results2)    
    return results2


job_details = process_job_data(job_datas)


print(job_details)


df = pd.DataFrame(job_details)

# Display the DataFrame
print(df)
df.to_csv('job_details2.csv')





params = {
  "api_key": "aa52dda07aca37c1537058311ddea2dedf13c06a954dc05af4aa866a66a61b28",
  "engine": "google_jobs",
  "google_domain": "google.com",
  "q": keywords,
  "gl": "in",
  "google_domain": "google.com",
  "num": "60",
  "start": "0",
  "location": location_name,
  "country": location_name,
  
}

search = GoogleSearch(params)
results = search.get_dict()
print(results)


def get_job_details(api_response):
    job_details2 = []

    for job in api_response['jobs_results']:
        title = job['title']
        company = job['company_name']
        job_link = job.get('job_link', 'Link not provided')
        job_description = job['description']
        job_id = job.get('job_id', 'ID not provided')
        related_links = job.get('related_links', 'Related links not provided')

        skills_required = []

        # Extracting skills from job highlights
        job_highlights = job['job_highlights']
        for highlight in job_highlights:
            items = highlight['items']
            for item in items:
                if 'https' not in item and 'Skills' in item:
                    skills = item.split(':')[1].strip()
                    skills_required.append(skills)

        job_details2.append({
            'title': title,
            'company': company,
            'job_link': job_link,
            'job_description': job_description,
            'job_id': job_id,
            'related_links': related_links,
            'skills_required': skills_required
        })

    return job_details2

# API response data (simplified for clarity)



final_results = get_job_details(results)
print(final_results)

len(final_results)


description_list = [] 
for i in range(len(final_results)):
    description_list.append(final_results[i]['job_description'])
    
print(description_list)






def extract_skills(job_description):
    # List of common technical skills and keywords
    tech_skills = set([
    'python', 'java', 'c++', 'javascript', 'sql', 'nosql', 'aws', 'azure', 'gcp',
    'docker', 'kubernetes', 'machine learning', 'deep learning', 'ai', 'data science',
    'data analysis', 'data engineering', 'big data', 'hadoop', 'spark', 'tableau',
    'power bi', 'excel', 'r', 'scala', 'git', 'devops', 'ci/cd', 'agile', 'scrum',
    'rest api', 'microservices', 'cloud computing', 'database', 'networking',
    'security', 'linux', 'windows', 'mac os', 'ios', 'android', 'react', 'angular',
    'vue', 'node.js', 'express', 'django', 'flask', 'spring', 'hibernate', 'orm',
    'api', 'json', 'xml', 'html', 'css', 'sass', 'less', 'webpack', 'babel',
    'typescript', 'redux', 'graphql', 'rest', 'soap', 'mvc', 'mvvm', 'tdd', 'cicd',
    'jenkins', 'travis ci', 'circle ci', 'terraform', 'ansible', 'puppet', 'chef',
    'mongodb', 'postgresql', 'mysql', 'oracle', 'ms sql server', 'redis', 'elasticsearch',
    'kafka', 'rabbitmq', 'activemq', 'tcp/ip', 'http', 'https', 'ssl/tls', 'oauth',
    'saml', 'ldap', 'active directory', 'unix', 'shell scripting', 'powershell', 'bash',
    'perl', 'ruby', 'go', 'rust', 'kotlin', 'swift', 'objective-c', 'xamarin',
    'react native', 'flutter', 'unity', 'unreal engine', 'photoshop', 'illustrator',
    'indesign', 'sketch', 'figma', 'invision', 'zeplin', 'jira', 'confluence', 'trello',
    'asana', 'slack', 'microsoft teams', 'zoom', 'webex', 'kanban', 'lean', 'six sigma',
    'prince2', 'pmp', 'itil', 'cobit', 'togaf', 'uml', 'erd', 'data modeling', 'etl',
    'olap', 'oltp', 'data warehouse', 'data lake', 'business intelligence',
    'data visualization', 'predictive analytics', 'statistical analysis', 'a/b testing',
    'seo', 'sem', 'google analytics', 'digital marketing', 'content management', 'crm',
    'erp', 'sap', 'salesforce', 'dynamics 365', 'workday', 'netsuite', 'blockchain',
    'cryptocurrency', 'iot', 'augmented reality', 'virtual reality', 'computer vision',
    'natural language processing', 'reinforcement learning', 'generative ai',
    'cybersecurity', 'penetration testing', 'ethical hacking', 'cryptography', '5g',
    'wifi', 'bluetooth', 'rfid', 'nfc', 'quantum computing', 'fpga', 'embedded systems',
    'robotics', 'plc', 'scada', 'can bus', 'modbus', 'cloud architecture', 'lambda',
    'cloudformation', 'docker swarm', 'helm', 'prometheus', 'grafana', 'elk stack',
    'logstash', 'graylog', 'splunk', 'new relic', 'datadog', 'opentracing', 'openmetrics',
    'loggly', 'sumologic', 'appdynamics', 'dynatrace', 'servicenow', 'incident management',
    'chaos engineering', 'site reliability engineering', 'bluemix', 'openshift', 'cloud foundry',
    'cloudbees', 'spinnaker', 'terraform cloud', 'packer', 'vault', 'consul', 'nomad',
    'vagrant', 'artifactory', 'nexus', 'sonarQube', 'veracode', 'snyk', 'blackduck', 'twistlock',
    'aqua', 'anchore', 'clair', 'trivy', 'aws lambda', 'aws ec2', 'aws s3', 'aws rds',
    'aws dynamodb', 'aws sagemaker', 'aws emr', 'azure functions', 'azure devops',
    'azure active directory', 'azure sql database', 'azure cosmos db', 'gcp functions',
    'gcp bigquery', 'gcp datastore', 'gcp pubsub', 'gcp firestore', 'gcp cloud run',
    'cloud storage', 'cloud sql', 'cloud spanner', 'bigtable', 'dataflow', 'dataproc',
    'firestore', 'pubsub', 'composer', 'cloud endpoints', 'cloud iam', 'cloud kms',
    'cloud scheduler', 'cloud tasks', 'cloud vision', 'cloud speech', 'cloud translate',
    'autoML', 'vertex ai', 'vertex pipelines', 'airflow', 'kubeflow', 'mlflow',
    'h2o.ai', 'datarobot', 'aws cloudtrail', 'aws config', 'aws guardduty', 'aws inspector',
    'aws security hub', 'aws shield', 'aws wafv2', 'aws sso', 'aws secrets manager',
    'azure security center', 'azure sentinel', 'azure policy', 'azure blueprints',
    'gcp security command center', 'gcp cloud security scanner', 'gcp vpc service controls',
    'gcp identity-aware proxy', 'gcp security key management', 'gcp confidential computing',
    'network security', 'application security', 'endpoint security', 'data security',
    'identity and access management', 'incident response', 'threat hunting',
    'vulnerability management', 'cyber threat intelligence', 'forensic analysis',
    'compliance and regulation', 'gdpr', 'ccpa', 'pci dss', 'hipaa', 'soc 2', 'iso 27001',
    'nist', 'csf', 'dfars', 'fedramp', 'cloud security', 'devsecops', 'zero trust architecture',
    'sase', 'sd-wan', 'vpn', 'firewall', 'ids/ips', 'siem', 'soar', 'xdr', 'cspm',
    'cdp', 'msp', 'mdm', 'uem', 'pam', 'waf', 'dos/ddos protection', 'network segmentation',
    'microsegmentation', 'end-to-end encryption', 'homomorphic encryption',
    'post-quantum cryptography', 'threat modeling', 'attack surface management',
    'red teaming', 'blue teaming', 'purple teaming', 'bug bounty', 'ethical hacking',
    'penetration testing', 'offensive security', 'defensive security', 'reverse engineering',
    'malware analysis', 'exploit development', 'incident response', 'digital forensics',
    'root cause analysis', 'threat hunting', 'security operations', 'security monitoring',
    'security analytics', 'security architecture', 'security engineering', 'security automation',
    'security orchestration', 'security testing', 'security auditing', 'security training',
    'awareness training', 'phishing simulation', 'cyber hygiene', 'security metrics',
    'security reporting', 'security compliance', 'security governance', 'security risk management',
    'business continuity', 'disaster recovery', 'crisis management', 'incident management',
    'data loss prevention', 'identity governance', 'threat intelligence platforms',
    'threat hunting platforms', 'security data lakes', 'security data fabric', 'security data mesh',
    'observability', 'monitoring', 'logging', 'tracing', 'alerting', 'cloud cost management',
    'finops', 'cloud billing', 'cloud cost optimization', 'cloud cost monitoring', 'cost explorer',
    'cloud cost governance', 'cloud cost allocation', 'cloud cost control', 'cloud cost transparency',
    'cloud cost visualization', 'cloud cost benchmarking', 'cloud cost forecasting', 'cloud cost savings',
    'cloud cost efficiency', 'cloud cost management tools', 'cloud cost management best practices',
    'cloud cost management strategies'])

    # Preprocess the job description
    job_description = job_description.lower()
    job_description = re.sub(r'[^\w\s]', '', job_description)

    # Tokenize the job description
    tokens = nltk.word_tokenize(job_description)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    # Extract single-word skills
    single_word_skills = set(tokens) & tech_skills

    # Extract multi-word skills
    vectorizer = CountVectorizer(vocabulary=tech_skills, ngram_range=(2,3))
    multi_word_skills = vectorizer.fit_transform([job_description]).nonzero()[1]
    multi_word_skills = [vectorizer.get_feature_names_out()[i] for i in multi_word_skills]

    # Combine single-word and multi-word skills
    extracted_skills = list(single_word_skills) + multi_word_skills

    return extracted_skills


def extract_skills_from_descriptions(description_list):
    skills_list = []
    for description in description_list:
        skills = extract_skills(description)
        skills_list.append(skills)
    return skills_list


skills_list = extract_skills_from_descriptions(description_list)

# Print extracted skills for each job description
for i, skills2 in enumerate(skills_list, 1):
    print(f"Skills for Job Description {i}:")
    print(skills2)
    print()


print(skills_list)

for job in results['jobs_results']:
    title = job['title']
    company = job['company_name']
    link = job['related_links']
    job_description = job['description']
    job_id = job['job_id']
    job_link = job.get('job_link', 'Link not provided')
    print(f"Job Title: {title}\nCompany: {company}\nJob Link: {link}\nJob Description: {job_description}\nJob ID: {job_id}")
    
    # Extracting skills from job highlights
    
    





if len(final_results) != len(skills_list):
    print("Warning: The number of job descriptions doesn't match the number of skill lists.")
    print(f"Number of job descriptions: {len(final_results)}")
    print(f"Number of skill lists: {len(skills_list)}")



# Append skills to each job description in final_results
for i, skills in enumerate(skills_list):
    if i < len(final_results):
        final_results[i]['extracted_skills'] = skills



# Print the updated final_results to verify
for i, job in enumerate(final_results):
    print(f"Job {i + 1}:")
    print(f"Title: {job.get('title', 'N/A')}")
    print(f"Company: {job.get('company', 'N/A')}")
    print(f"Extracted Skills: {job.get('extracted_skills', [])}")
    print()

job_data2 = pd.DataFrame(final_results)
print(job_data2)

print(skills2)






def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# User skills
user_skills = [skill.strip() for skill in skills2.split(',')]

# Combine all skills into a single string
user_skills_text = ", ".join(user_skills)

def calculate_job_match(row):
    # Combine title and skills (if available) for job text
    job_text = row['title']
    if isinstance(row['extracted_skills'], str) and row['extracted_skills'] != '[]':
        job_text += " " + row['extracted_skills']
    
    # Calculate similarity
    payload = {
        "inputs": {
            "source_sentence": user_skills_text,
            "sentences": [job_text]
        }
    }
    result = query(payload)
    
    # The API returns a list of similarities, we only sent one job so we take the first (and only) result
    return result[0] if isinstance(result, list) and len(result) > 0 else 0

# Calculate match scores for all jobs
job_data2['match_score'] = job_data2.apply(calculate_job_match, axis=1)

# Sort jobs by match score in descending order
top_jobs = job_data2.sort_values('match_score', ascending=False).head(5)

# Print results
for _, job in top_jobs.iterrows():
    print(f"Title: {job['title']}")
    print(f"Match score: {job['match_score']:.2f}")
    print(f"Company: {job['company']}")
    print(f"Skills: {job['extracted_skills']}")
    print(f"Job_id: {job['job_id']}")
    print()


results3 = []
for _, job in top_jobs.iterrows():
    job_result = {
        "title": job['title'],
        "match_score": job['match_score'],
        "company": job['company'],
        "skills": job['extracted_skills'],
        "job_id": job['job_id']
        
    }
    results3.append(job_result)

# Return results (you can further process or display these as needed)
print(results3)


job_ids2 = [job['job_id'] for job in results3]
company_name = [job['company'] for job in results3]

print(job_ids2)


print(company_name)


def get_job_details2(job_id):
 
 
  params = {
  "api_key": "aa52dda07aca37c1537058311ddea2dedf13c06a954dc05af4aa866a66a61b28",
  "engine": "google_jobs_listing",
  "q": job_id
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  return results


def get_job2(job_ids2):
    job_datas2 = []
    for job_id in job_ids2:
        job_data = get_job_details2(job_id)
        job_datas2.append(job_data)
        
        print(f"Iteration {len(job_datas2)}: job_id={job_id}, job_data={job_data}")

    

    return job_datas2


job_datas2= get_job2(job_ids2)
print(job_datas2)


print(job_datas2)



def extract_job_details2(api_response):
    # Extract job title
    title = api_response.get('search_parameters', {}).get('q', '')
    
    # Extract company name and apply link (if available)
    
    apply_link = None
    if 'apply_options' in api_response and api_response['apply_options']:
        first_option = api_response['apply_options'][0]
        
        apply_link = first_option['link']
    
    # Extract job location (not available in this API response)
    location = "Not specified"
    
    # Extract job description (not directly available in this API response)
    description = "Full job description not available in this API response."
    
    # Extract salary information
    salaries = api_response.get('salaries', [])
    salary_info = []
    for salary in salaries:
        salary_info.append({
            'job_title': salary.get('job_title'),
            'salary_from': salary.get('salary_from'),
            'salary_to': salary.get('salary_to'),
            'salary_period': salary.get('salary_period'),
            'source': salary.get('source')
        })
    
    # Extract apply options
    apply_options = [{'title': option['title'], 'link': option['link']} 
                     for option in api_response.get('apply_options', [])]
    
    # Extract company ratings
    ratings = [{'source': rating['source'], 'rating': rating['rating'], 'reviews': rating['reviews']} 
               for rating in api_response.get('ratings', [])]
    
    # Prepare job details object
    job_details = {
        'job_title': title,
        
        'location': location,
        'job_description': description,
        'apply_link': apply_link,  # Added apply link here
        'salary_info': salary_info,
        'apply_options': apply_options,
        'company_ratings': ratings
    }
    
    return job_details


def process_job_data2(job_data_list):
    results2 = []
    for job_data in job_data_list:
        job_details = extract_job_details2(job_data)
        results2.append(job_details)
        print(f"Iteration {len(results2)}: job_details={job_details}")
    print(results2)    
    return results2

job_details2=process_job_data2(job_datas2)
print(job_details2)


df2 = pd.DataFrame(job_details2)

# Display the DataFrame


for i in range(len(df2)):
    df2.loc[i, 'company'] = company_name[i]

print(df2)
df2.to_csv('job_details2.3.csv')




