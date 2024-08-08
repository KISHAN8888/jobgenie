import re
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords

import nltk

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

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

# Example usage
job_description = """
'Exploring and acquiring new sources of data. Using curiosity and creativity, would involve using technology to automate data acquisition, the ability to link new sources with internal data, and an appropriate level of supporting documentation of the source and technical solution. Ensuring quality and understanding of analytical data. This would require hands-on data experience to investigate data issues and seek to resolution or acceptance. Create the appropriate amount of documentation, leverage standards and build upon them. Data should be reconciled and documented at various stages for integrity. Participate in developing governance and rigour of data management practice within the Data Sourcing Solutions Team . This will also include partnering with enterprise IT groups and involvement in enterprise data-related functions. You will report to the Data Engineering Manager/Principal Data Engineer. SHARE your talent We re looking for someone who has these abilities and skills... Required Skills and Abilities: Relevant years of experience is required. Sound understanding/experience of Python, Spark, Databricks PySpark/Spark SQL and best practices Effective SQL Skills knowledge is required. Data development experience using data development / Analysis tools. Expertise in Star Schema data modelling. Experience working on building /optimizing logical data models and data pipelines while delivering high-data-quality solutions that are testable and adhere to SLAs. Development experience using R(Data structures, Plumber, Shiny) a plus point. Familiarity with developer support tools (TFS/GIT, Jenkins) a plus. Demonstrated ability to work through data complexities which include a variety of sources, formats, and structures. Desired Skills and Abilities: Data Engineering and Insurance background preferred. Effective preference of experience in the Insurance domain. Ability to see through ambiguous concepts, and break down complex problems into manageable components. Detail-orientated, proven ability to recognize patterns in data. Effective analytical and problem-solving skills. Possesses natural curiosity. Seek to understand the world around you, and question when appropriate. College Degree in MIS, Information Technology, Computer Science, Engineering, Statistics, Mathematics, Actuarial Science or equivalent', 'job_highlights': [{'items': ['Exploring and acquiring new sources of data. Using curiosity and creativity, would involve using technology to automate data acquisition, the ability to link new sources with internal data, and an appropriate level of supporting documentation of the source and technical solution. Ensuring quality and understanding of analytical data. This would require hands-on data experience to investigate data issues and seek to resolution or acceptance. Create the appropriate amount of documentation, leverage standards and build upon them. Data should be reconciled and documented at various stages for integrity. Participate in developing governance and rigour of data management practice within the Data Sourcing Solutions Team . This will also include partnering with enterprise IT groups and involvement in enterprise data-related functions. You will report to the Data Engineering Manager/Principal Data Engineer. SHARE your talent We re looking for someone who has these abilities and skills... Required Skills and Abilities: Relevant years of experience is required. Sound understanding/experience of Python, Spark, Databricks PySpark/Spark SQL and best practices Effective SQL Skills knowledge is required. Data development experience using data development / Analysis tools. Expertise in Star Schema data modelling. Experience working on building /optimizing logical data models and data pipelines while delivering high-data-quality solutions that are testable and adhere to SLAs. Development experience using R(Data structures, Plumber, Shiny) a plus point. Familiarity with developer support tools (TFS/GIT, Jenkins) a plus. Demonstrated ability to work through data complexities which include a variety of sources, formats, and structures. Desired Skills and Abilities: Data Engineering and Insurance background preferred. Effective preference of experience in the Insurance domain. Ability to see through ambiguous concepts, and break down complex problems into manageable components. Detail-orientated, proven ability to recognize patterns in data. Effective analytical and problem-solving skills. Possesses natural curiosity. Seek to understand the world around you, and question when appropriate. College Degree in MIS, Information Technology, Computer Science, Engineering, Statistics, Mathematics, Actuarial Science or equivalent'
"""

extracted_skills = extract_skills(job_description)
print("Extracted skills:", extracted_skills)