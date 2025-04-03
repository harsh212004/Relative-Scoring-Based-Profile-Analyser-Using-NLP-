import streamlit as st
import pandas as pd
import base64, random
import time,datetime
import pymysql
import os
import socket
import platform
import geocoder
import secrets
import io,random
import plotly.express as px
import plotly.graph_objects as go
from geopy.geocoders import Nominatim
# libraries used to parse the pdf files
from pyresparser import ResumeParser
from pdfminer3.layout import LAParams
from pdfminer3.pdfpage import PDFPage
from pdfminer3.pdfinterp import PDFResourceManager
from pdfminer3.pdfinterp import PDFPageInterpreter
from pdfminer3.converter import TextConverter
from streamlit_tags import st_tags
from PIL import Image
# pre stored data for prediction purposes
from Courses import ds_course,webdev_course,android_course,ios_course,uiux_course,resume_videos,interview_videos
import nltk
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
nltk.download('stopwords')


###### Preprocessing functions ######
class SkillGapAnalyzer:
    def __init__(self):
        self.job_profiles = {
            'Data Science': {
                'required_skills': ['python', 'machine learning', 'deep learning', 'statistics', 
                                   'data analysis', 'sql', 'pandas', 'numpy', 'matplotlib', 
                                   'seaborn', 'tensorflow', 'pytorch', 'scikit-learn', 
                                   'data visualization', 'data cleaning', 'natural language processing',
                                   'nlp', 'computer vision', 'big data', 'Web Coding', 'Microsoft Word', 'Excel', 'Powerpoint'
                                   'data mining', 'predictive modeling', 'statistical analysis', 'Data Analysis', 'AI Tool Development', 'data processing systems',
                                   'feature engineering', 'model deployment'],
                'keywords': ['tensorflow', 'pytorch', 'machine learning', 'deep learning', 
                           'data science', 'python', 'analytics', 'scikit-learn', 'pandas', 
                           'numpy', 'matplotlib', 'seaborn', 'nlp', 'computer vision', 
                           'sql', 'eda', 'mlops', 'reinforcement learning', 'ai', 'artificial intelligence',
                           'neural networks', 'data pipeline', 'jupyter', 'colab'],
                'descriptions': [
                    "Entry-level data science role ideal for third or final year students with knowledge of Python, statistics, and data analysis.",
                    "Machine learning internship requiring familiarity with TensorFlow/PyTorch, model deployment, and basic deep learning concepts.",
                    "Data analyst trainee position focused on SQL, data visualization, and business intelligence for real-world applications."
                ]
            },
            'Web Development': {
                'required_skills': ['javascript', 'html', 'css', 'react', 'angular', 
                                  'vue', 'node.js', 'express', 'django', 'flask', 
                                  'php', 'mongodb', 'mysql', 'postgresql', 'rest api', 'restapi',
                                  'git', 'github', 'responsive design', 'web security',
                                  'aws', 'azure', 'typescript'],
                'keywords': ['react', 'django', 'node.js', 'react.js', 'php', 'laravel', 
                           'javascript', 'angular', 'c#', 'flask', 'express.js', 
                           'mongodb', 'mysql', 'rest api', 'restapi', 'full-stack', 'frontend',
                           'backend', 'web development', 'html5', 'css3', 'bootstrap'],
                'descriptions': [
                    "Frontend developer internship for students with HTML, CSS, JavaScript, and modern frameworks like React or Angular.",
                    "Full-stack development opportunity requiring proficiency in Node.js and database management, ideal for final-year projects.",
                    "Backend development role focused on API creation, database handling, and security principles for web applications."
                ]
            },
            'Android Development': {
                'required_skills': ['java', 'kotlin', 'android studio', 'xml', 
                                   'firebase', 'mvvm', 'dagger', 'retrofit', 
                                   'room', 'jetpack', 'material design', 'android sdk',
                                   'git', 'github', 'debugging', 'performance optimization',
                                   'unit testing', 'ui/ux', 'workmanager', 'picasso'],
                'keywords': ['android', 'android development', 'flutter', 'kotlin', 
                           'xml', 'kivy', 'java', 'mobile development', 'android studio',
                           'jetpack compose', 'android jetpack', 'material design', 'mobile app',
                           'google play', 'play store', 'apk', 'sdk', 'ndk', 'android architecture'],
                'descriptions': [
                    "Android development internship requiring Java/Kotlin knowledge and Android SDK experience, suitable for students building mobile apps.",
                    "Mobile app development trainee role focusing on Jetpack, MVVM, and Firebase integration for real-world applications.",
                    "Cross-platform development position emphasizing Flutter or React Native for students exploring mobile frameworks."
                ]
            },
            'iOS Development': {
                'required_skills': ['swift', 'objective-c', 'xcode', 'cocoa touch',
                                   'swiftui', 'uikit', 'core data', 'core animation',
                                   'autolayout', 'git', 'github', 'mvvm', 'viper',
                                   'alamofire', 'combine', 'unit testing', 'app store'],
                'keywords': ['ios', 'ios development', 'swift', 'objective-c', 
                           'xcode', 'cocoa touch', 'mobile development', 'apple',
                           'app store', 'testflight', 'arkit', 'core ml', 'swiftui',
                           'uikit', 'watchos', 'tvos', 'ipados', 'mac catalyst'],
                'descriptions': [
                    "iOS development internship for students skilled in Swift and UIKit, ideal for final-year mobile projects.",
                    "Mobile development trainee role focusing on SwiftUI and core Apple frameworks for app development.",
                    "iOS engineer position emphasizing performance optimization, App Store deployment, and UI/UX best practices."
                ]
            },
            'UI/UX Development': {
                'required_skills': ['Figma', 'adobe xd', 'sketch', 'illustrator',
                                   'photoshop', 'user research', 'wireframing',
                                   'prototyping', 'usability testing', 'color theory',
                                   'typography', 'design systems', 'responsive design',
                                   'accessibility', 'interaction design', 'animation',
                                   'after effects', 'user flows', 'information architecture',
                                   'user personas'],
                'keywords': ['ux', 'ui', 'user experience', 'user interface', 'adobe xd', 'Figma', 'zeplin', 'balsamiq', 
                            'ui design', 'ux design', 'prototyping', 'wireframes', 'storyframes', 
                            'photoshop', 'illustrator', 'after effects', 'premier pro', 'indesign', 
                            'user research', 'usability testing', 'design system', 'material design',
                            'human interface', 'design sprint', 'user centered design', 'interaction design',
                            'visual design', 'motion design', 'design tools', 'ui toolkit'],
                'descriptions': [
                    "UI/UX internship focusing on Figma, wireframing, and prototyping for students building design portfolios.",
                    "User research trainee position requiring usability testing and user-centered design knowledge, ideal for final-year students.",
                    "Product design internship emphasizing interaction design, visual storytelling, and accessibility principles for practical applications."
                ]
            }
        }
        
        self._prepare_analyzer()

        self._prepare_semantic_matchers()

    def _prepare_semantic_matchers(self):
        """Prepare optimized semantic matching components"""
        # Create corpus focusing only on required skills
        corpus = []
        for profile in self.job_profiles.values():
            corpus.extend(profile['required_skills'])
            corpus.extend(profile['keywords'])
        
        # Optimized TF-IDF Vectorizer
        self.tfidf_vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8  # Ignore terms that appear in more than 80% of documents
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(corpus)
        
        # Lightweight Word2Vec Model with skill-focused training
        tokenized_corpus = [[skill.lower()] for skill in corpus] 
        self.word2vec_model = Word2Vec(
            tokenized_corpus,
            vector_size=50, 
            window=3,       
            min_count=1,
            workers=4,
            epochs=20   
        )
    
    def _get_semantic_similarity(self, skill1, skill2):
        """Combined semantic similarity score with optimized weights"""
        # Exact match shortcut
        if skill1.lower() == skill2.lower():
            return 1.0
            
        # TF-IDF similarity
        try:
            tfidf_sim = self._get_tfidf_similarity(skill1, skill2)
        except:
            tfidf_sim = 0
            
        # Word2Vec similarity
        try:
            w2v_sim = self._get_word2vec_similarity(skill1, skill2)
        except:
            w2v_sim = 0
            
        # Combined score with more weight to TF-IDF
        return (0.7 * tfidf_sim) + (0.3 * w2v_sim)
    
    def _get_tfidf_similarity(self, skill1, skill2):
        """Optimized TF-IDF similarity calculation"""
        vectors = self.tfidf_vectorizer.transform([skill1, skill2])
        return cosine_similarity(vectors[0], vectors[1])[0][0]
    
    def _get_word2vec_similarity(self, skill1, skill2):
        """Optimized Word2Vec similarity calculation"""
        def get_skill_vector(skill):
            words = skill.lower().split()
            valid_vectors = [self.word2vec_model.wv[word] for word in words 
                           if word in self.word2vec_model.wv]
            if not valid_vectors:
                return None
            return np.mean(valid_vectors, axis=0)
            
        vec1 = get_skill_vector(skill1)
        vec2 = get_skill_vector(skill2)
        
        if vec1 is None or vec2 is None:
            return 0
            
        return cosine_similarity([vec1], [vec2])[0][0]
    
    def _match_skills(self, resume_skills, job_skills):
        """Optimized skill matching with hybrid approach"""
        matched = set()
        exact_match_threshold = 0.9
        partial_match_threshold = 0.7
        
        for r_skill in resume_skills:
            for j_skill in job_skills:
                # First check exact match (fastest)
                if r_skill.lower() == j_skill.lower():
                    matched.add(j_skill)
                    break
                    
                # Then check partial string match
                if (r_skill.lower() in j_skill.lower() or 
                    j_skill.lower() in r_skill.lower()):
                    matched.add(j_skill)
                    break
                
                # Finally use semantic matching if above thresholds aren't met
                sim_score = self._get_semantic_similarity(r_skill, j_skill)
                if sim_score >= exact_match_threshold:
                    matched.add(j_skill)
                    break
                elif sim_score >= partial_match_threshold:
                    # Only add if not already matched by another method
                    if j_skill not in matched:
                        matched.add(j_skill)
                        break
        
        return matched
    
    def _prepare_analyzer(self):
        """Prepare all necessary components for analysis"""
        all_skills = set()
        for profile in self.job_profiles.values():
            all_skills.update(profile['required_skills'])
        self.skill_vocabulary = list(all_skills)
        
        for profile_name, profile in self.job_profiles.items():
            profile['skill_vector'] = np.array([
                1 if skill in profile['required_skills'] else 0 
                for skill in self.skill_vocabulary
            ])
    
    def _preprocess_skills(self, skills):
        """Enhanced skill normalization with better handling of variations"""
        processed = []
        skill_variations = {
            'python': ['python', 'py', 'python3', 'python 3'],
            'machine learning': ['ml', 'machine learning', 'machine-learning'],
            'deep learning': ['dl', 'deep learning', 'deep-learning'],
            'data analysis': ['data analysis', 'data analytics'],
            'sql': ['sql', 'structured query language'],
            'pandas': ['pandas', 'pd'],
            'numpy': ['numpy', 'np'],
            'matplotlib': ['matplotlib', 'plt', 'mpl'],
            'seaborn': ['seaborn', 'sns'],
            'tensorflow': ['tensorflow', 'tf'],
            'pytorch': ['pytorch', 'torch'],
            'scikit-learn': ['scikit-learn', 'sklearn', 'scikit learn'],
            'natural language processing': ['nlp', 'natural language processing'],
            'javascript': ['javascript', 'js', 'ecmascript'],
            'html': ['html', 'html5'],
            'css': ['css', 'css3'],
            'react': ['react', 'react.js', 'reactjs'],
            'angular': ['angular', 'angular.js', 'angularjs'],
            'vue': ['vue', 'vue.js', 'vuejs'],
            'node.js': ['node', 'node.js', 'nodejs'],
            'express': ['express', 'express.js', 'expressjs'],
            'django': ['django', 'django framework'],
            'flask': ['flask', 'flask framework'],
            'mongodb': ['mongodb', 'mongo'],
            'mysql': ['mysql', 'my sql'],
            'postgresql': ['postgresql', 'postgres', 'postgres db'],
            'rest api': ['rest', 'rest api', 'restapi', 'restful api'],
            'git': ['git', 'git version control'],
            'github': ['github', 'git hub'],
            'docker': ['docker', 'docker container'],
            'kubernetes': ['kubernetes', 'k8s'],
            'java': ['java', 'java programming'],
            'kotlin': ['kotlin', 'kt'],
            'android studio': ['android studio', 'android ide'],
            'swift': ['swift', 'swift programming'],
            'objective-c': ['objective-c', 'objc', 'objective c'],
            'xcode': ['xcode', 'xcode ide'],
            'figma': ['Figma', 'Figma design'],
            'adobe xd': ['adobe xd', 'xd'],
            'sketch': ['sketch', 'sketch app'],
            'photoshop': ['photoshop', 'ps', 'adobe photoshop'],
            'illustrator': ['illustrator', 'ai', 'adobe illustrator'],
            'aws': ['aws', 'amazon web services'],
            'azure': ['azure', 'microsoft azure']
        }
        
        # Reverse mapping for faster lookup
        variation_to_standard = {}
        for std_skill, variations in skill_variations.items():
            for variation in variations:
                variation_to_standard[variation] = std_skill
        
        for skill in skills:
            skill = str(skill).lower().strip()
            skill = re.sub(r'[^a-zA-Z0-9\s/+]', '', skill)
            
            if skill in variation_to_standard:
                processed.append(variation_to_standard[skill])
                continue
                
            # Handle compound skills with slashes (like UI/UX)
            if '/' in skill:
                parts = [p.strip() for p in skill.split('/')]
                matched_parts = []
                for part in parts:
                    if part in variation_to_standard:
                        matched_parts.append(variation_to_standard[part])
                    else:
                        for std_skill in self.skill_vocabulary:
                            if self._calculate_similarity(part, std_skill) > 0.8:
                                matched_parts.append(std_skill)
                                break
                
                if matched_parts:
                    processed.extend(matched_parts)
                else:
                    processed.append(skill)
                continue
                
            # Handle skills with plus (like C++)
            if '+' in skill:
                processed.append(skill.replace('+', ' plus ').strip())
                continue
                
            found = False
            for std_skill in self.skill_vocabulary:
                if self._calculate_similarity(skill, std_skill) > 0.8:
                    processed.append(std_skill)
                    found = True
                    break
                    
            if not found:
                processed.append(skill)
                
        return list(set(processed))  # Remove duplicates
    
    def _calculate_similarity(self, word1, word2):
        """Calculate similarity between two words using Levenshtein distance"""
        from Levenshtein import ratio
        return ratio(word1, word2)
    
    def _match_skills(self, resume_skills, job_skills):
        """Enhanced skill matching with fuzzy matching and partial matching"""
        matched = set()
        
        for r_skill in resume_skills:
            for j_skill in job_skills:
                # Exact match
                if r_skill == j_skill:
                    matched.add(j_skill)
                    break
                
                # Partial match (e.g., 'ui' matches 'ui/ux')
                if r_skill in j_skill or j_skill in r_skill:
                    matched.add(j_skill)
                    break
                    
                # Fuzzy match
                if self._calculate_similarity(r_skill, j_skill) > 0.85:
                    matched.add(j_skill)
                    break
                    
                # Handle compound skills (e.g., 'ui/ux' vs 'ui' or 'ux')
                if '/' in r_skill or '/' in j_skill:
                    r_parts = r_skill.split('/')
                    j_parts = j_skill.split('/')
                    if any(rp in j_parts for rp in r_parts) or any(jp in r_parts for jp in j_parts):
                        matched.add(j_skill)
                        break
        return matched
    
    def analyze_gaps(self, resume_skills, target_job):
        """Analyze skill gaps between resume and target job with enhanced reporting"""
        try:
            if target_job not in self.job_profiles:
                return None
            resume_skills = self._preprocess_skills(resume_skills)
            
            # Get target job requirements
            job_profile = self.job_profiles[target_job]
            job_skills = job_profile['required_skills']
            job_keywords = job_profile['keywords']

            matched_skills = self._match_skills(resume_skills, job_skills)
            resume_vector = np.array([
                1 if skill in matched_skills else 0 
                for skill in self.skill_vocabulary
            ])
            
            job_vector = job_profile['skill_vector']
            
            # Calculate match percentage
            matching_skills = np.sum(resume_vector & job_vector)
            total_required = np.sum(job_vector)
            match_percentage = (matching_skills / total_required) * 100 if total_required > 0 else 0
            
            # Identify missing skills
            missing_indices = np.where((job_vector - resume_vector) > 0)[0]
            missing_skills = [self.skill_vocabulary[i] for i in missing_indices]
            
            # Get top 5 most important missing skills (prioritize skills that appear first in required_skills)
            required_order = {skill: idx for idx, skill in enumerate(job_profile['required_skills'])}
            missing_skills = sorted(
                missing_skills,
                key=lambda x: required_order.get(x, len(job_profile['required_skills'])))
            missing_skills = missing_skills[:5]
            
            # Get user's existing skills that match the profile (top 5 most relevant)
            user_skills = list(matched_skills)
            user_skills = sorted(
                user_skills,
                key=lambda x: required_order.get(x, len(job_profile['required_skills'])))
            user_skills = user_skills[:5]

            matched_keywords = self._match_skills(resume_skills, job_keywords)
            keyword_coverage = (len(matched_keywords) / len(job_keywords)) * 100 if job_keywords else 0
            
            return {
                'match_percentage': round(match_percentage, 2),
                'keyword_coverage': round(keyword_coverage, 2),
                'missing_skills': missing_skills,
                'top_skills': job_profile['required_skills'][:10],
                'job_descriptions': job_profile['descriptions'],
                'your_skills': user_skills,
                'matched_keywords': matched_keywords,
                'all_required_skills': job_profile['required_skills'],
                'resume_skills': resume_skills
            }
            
        except Exception as e:
            st.error(f"Skill gap analysis error: {str(e)}")
            return None
    
def validate_name(name):
    return bool(re.match(r"^[A-Za-z\s]+$", name))

def validate_email(email):
    return bool(re.match(r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$", email))

def validate_phone(phone):
    return bool(re.match(r"^\d{10}$", phone))

def get_csv_download_link(df,filename,text):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()      
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">{text}</a>'
    return href

def extract_name(resume_text):
    import re
    name_pattern = re.compile(r'([A-Z][a-z]+(?:\s[A-Z][a-z]+)+)')
    match = name_pattern.search(resume_text)
    if match:
        return match.group(1)
    return None

def pdf_reader(file):
    resource_manager = PDFResourceManager()
    fake_file_handle = io.StringIO()
    converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
    page_interpreter = PDFPageInterpreter(resource_manager, converter)
    with open(file, 'rb') as fh:
        for page in PDFPage.get_pages(fh,
                                      caching=True,
                                      check_extractable=True):
            page_interpreter.process_page(page)
            print(page)
        text = fake_file_handle.getvalue()

    converter.close()
    fake_file_handle.close()
    return text


def show_pdf(file_path):
    with open(file_path, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)


def course_recommender(course_list):
    st.subheader("Courses & Certificates Recommendations")
    c = 0
    rec_course = []
    ## slider to choose from range 1-10
    no_of_reco = st.slider('Choose Number of Course Recommendations:', 1, 10, 5)
    random.shuffle(course_list)
    for c_name, c_link in course_list:
        c += 1
        st.markdown(f"({c}) [{c_name}]({c_link})")
        rec_course.append(c_name)
        if c == no_of_reco:
            break
    return rec_course


###### Database ######


connection = pymysql.connect(host='localhost',user='root',password='Harsh@2004',db='cv')
cursor = connection.cursor()


# inserting miscellaneous data, fetched results, prediction and recommendation into user_data table
def insert_data(sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country, 
                act_name, act_mail, act_mob, name, email, res_score, timestamp, no_of_pages, 
                skills, pdf_name):
    DB_table_name = 'user_data'
    insert_sql = f"""
    INSERT INTO {DB_table_name} 
    (sec_token, ip_add, host_name, dev_user, os_name_ver, latlong, city, state, country, 
     act_name, act_mail, act_mob, Name, Email_ID, resume_score, Timestamp, Page_no, 
     Predicted_Field, User_level, Actual_skills, Recommended_skills, Recommended_courses, pdf_name)
    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    rec_values = (
        str(sec_token), str(ip_add), host_name, dev_user, os_name_ver, str(latlong), 
        city, state, country, act_name, act_mail, act_mob, name, email, str(res_score), 
        timestamp, str(no_of_pages), '', '', str(skills), '', '', pdf_name
    )
    cursor.execute(insert_sql, rec_values)
    connection.commit()



# inserting feedback data into user_feedback table
def insertf_data(feed_name,feed_email,feed_score,comments,Timestamp):
    DBf_table_name = 'user_feedback'
    insertfeed_sql = "insert into " + DBf_table_name + """
    values (0,%s,%s,%s,%s,%s)"""
    rec_values = (feed_name, feed_email, feed_score, comments, Timestamp)
    cursor.execute(insertfeed_sql, rec_values)
    connection.commit()


###### Relative Scoring Functions ######


# Fetch all resume scores from the database
def fetch_all_resume_scores():
    query = "SELECT resume_score FROM user_data"
    cursor.execute(query)
    scores = cursor.fetchall()
    return [int(score[0]) for score in scores if score[0] is not None]

# Calculate the average resume score
def calculate_average_score(scores):
    if len(scores) == 0:
        return 0
    return sum(scores) / len(scores)


###### Setting Page Configuration (favicon, Logo, Title) ######


st.set_page_config(
   page_title="AI Resume Analyzer",
   page_icon='./Logo/recommend.png',
)

###### Main function run() ######

def run():
    img = Image.open('./Logo/logo3.png')
    st.image(img)

    # Custom CSS for Sidebar
    st.markdown(
    """
    <style>
    .sidebar .sidebar-content {
        width: 200px;  /* Adjust the width of the sidebar */
        font-size: 18px;  /* Increase the font size */
    }
    .sidebar .sidebar-content a {
        text-decoration: underline;  /* Underline the links */
        color: #021659;  /* Change the link color */
        display: block;  /* Make each link a block element */
        margin-bottom: 10px;  /* Add spacing between links */
    }
    .sidebar .sidebar-content a:hover {
        color: #d73b5c;  /* Change the link color on hover */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
    st.sidebar.markdown("# Navigation")
    st.sidebar.markdown("---") 

    activities = ["User", "Admin", "Feedback", "About"]
    choice = st.sidebar.selectbox("Choose an option:", activities, key='nav_select')


    ###### Creating Database and Table ######
    db_sql = """CREATE DATABASE IF NOT EXISTS CV;"""
    cursor.execute(db_sql)

    # Create table user_data and user_feedback
    DB_table_name = 'user_data'
    table_sql = "CREATE TABLE IF NOT EXISTS " + DB_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                    sec_token varchar(20) NOT NULL,
                    ip_add varchar(50) NULL,
                    host_name varchar(50) NULL,
                    dev_user varchar(50) NULL,
                    os_name_ver varchar(50) NULL,
                    latlong varchar(50) NULL,
                    city varchar(50) NULL,
                    state varchar(50) NULL,
                    country varchar(50) NULL,
                    act_name varchar(50) NOT NULL,
                    act_mail varchar(50) NOT NULL,
                    act_mob varchar(20) NOT NULL,
                    Name varchar(500) NOT NULL,
                    Email_ID VARCHAR(500) NOT NULL,
                    resume_score VARCHAR(8) NOT NULL,
                    Timestamp VARCHAR(50) NOT NULL,
                    Page_no VARCHAR(5) NOT NULL,
                    Predicted_Field BLOB NOT NULL,
                    User_level BLOB NOT NULL,
                    Actual_skills BLOB NOT NULL,
                    Recommended_skills BLOB NOT NULL,
                    Recommended_courses BLOB NOT NULL,
                    pdf_name varchar(50) NOT NULL,
                    PRIMARY KEY (ID)
                    );
                """
    cursor.execute(table_sql)

    DBf_table_name = 'user_feedback'
    tablef_sql = "CREATE TABLE IF NOT EXISTS " + DBf_table_name + """
                    (ID INT NOT NULL AUTO_INCREMENT,
                        feed_name varchar(50) NOT NULL,
                        feed_email VARCHAR(50) NOT NULL,
                        feed_score VARCHAR(5) NOT NULL,
                        comments VARCHAR(100) NULL,
                        Timestamp VARCHAR(50) NOT NULL,
                        PRIMARY KEY (ID)
                    );
                """
    cursor.execute(tablef_sql)
    

    ###### CODE FOR CLIENT SIDE (USER) ######


    if choice == 'User':
        act_name = st.text_input('Name*')
        act_mail = st.text_input('Mail*')
        act_mob = st.text_input('Mobile Number*')

        if act_name and not validate_name(act_name):
            st.error("Please enter a valid name (only letters and spaces allowed).")

        if act_mail and not validate_email(act_mail):
            st.error("Please enter a valid email address.")

        if act_mob and not validate_phone(act_mob):
            st.error("Please enter a valid 10-digit mobile number.")
            
        sec_token = secrets.token_urlsafe(12)
        host_name = socket.gethostname()
        ip_add = socket.gethostbyname(host_name)
        dev_user = os.getlogin()
        os_name_ver = platform.system() + " " + platform.release()
        g = geocoder.ip('me')
        latlong = g.latlng

        if latlong is not None:
            try:
                geolocator = Nominatim(user_agent="http")
                location = geolocator.reverse(latlong, language='en')
                
                if location and hasattr(location, 'raw') and location.raw is not None:
                    address = location.raw.get('address', {})
                    cityy = address.get('city', '')
                    statee = address.get('state', '')
                    countryy = address.get('country', '')
                else:
                    cityy = statee = countryy = ''
            except Exception as e:
                st.warning(f"Could not fetch location details: {str(e)}")
                cityy = statee = countryy = ''
        else:
            cityy = statee = countryy = ''

        city = cityy
        state = statee
        country = countryy

        st.markdown('''<h5 style='text-align: left; color: #021659;'> Upload Your Resume, And Get Smart Recommendations</h5>''',unsafe_allow_html=True)
        
        pdf_file = st.file_uploader("Choose your Resume", type=["pdf"])
        if pdf_file is not None:
            with st.spinner('Hang On While We Cook Magic For You...'):
                time.sleep(4)
        
            save_image_path = './Uploaded_Resumes/'+pdf_file.name
            pdf_name = pdf_file.name
            with open(save_image_path, "wb") as f:
                f.write(pdf_file.getbuffer())
            show_pdf(save_image_path)

            ### parsing and extracting whole resume 
            resume_data = ResumeParser(save_image_path).get_extracted_data()
            if resume_data:
                ## Get the whole resume data into resume_text
                resume_text = pdf_reader(save_image_path)

                ## Showing Analyzed data from (resume_data)
                st.header("Resume Analysis")

                st.subheader("Your Basic info")
                try:
                    if resume_data.get('name'):
                        st.text('Name: ' + resume_data['name'])
                    if resume_data.get('email'):
                        st.text('Email: ' + resume_data['email'])
                    if resume_data.get('mobile_number'):
                        st.text('Contact: ' + resume_data['mobile_number'])
                    if resume_data.get('degree'):
                        st.text('Degree: ' + str(resume_data['degree']))
                    if resume_data.get('no_of_pages'):
                        st.text('Resume pages: ' + str(resume_data['no_of_pages']))
                except Exception as e:
                    st.error(f"Error displaying basic info: {e}")

                # 2. Resume Score Section
                st.subheader("Resume Score")
                resume_score = 0

                ### Predicting Whether these key points are added to the resume
                # for objective or summary 6 points
                if 'Objective' in resume_text or 'Summary' in resume_text or 'About' in resume_text or 'About Me' in resume_text or 'serving' in resume_text:
                    resume_score = resume_score+6
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Objective/Summary</h4>''',unsafe_allow_html=True)                
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add your career objective, it will give your career intension to the Recruiters.</h4>''',unsafe_allow_html=True)
                
                # for education 12 points
                if 'Education' in resume_text or 'School' in resume_text or 'College' in resume_text or 'Qualifications' in resume_text or 'Qualification' in resume_text or 'qualification' in resume_text or 'qualifications' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Education Details</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Education. It will give Your Qualification level to the recruiter</h4>''',unsafe_allow_html=True)

                # Combined Experience/Internships check (20 points total)
                has_experience = False
                exp_keywords = ['EXPERIENCE', 'Experience', 'WORK EXPERIENCE', 'Work Experience', 
                            'INTERNSHIP', 'Internship', 'INTERNSHIPS', 'Internships']
                for keyword in exp_keywords:
                    if keyword in resume_text:
                        has_experience = True
                        break

                if has_experience:
                    resume_score = resume_score + 20
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Professional Experience/Internships</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Professional Experience or Internships. It will help you to stand out from crowd</h4>''',unsafe_allow_html=True)
                
                # for skills 7 points
                if 'SKILLS' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'SKILL' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skills' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                elif 'Skill' in resume_text:
                    resume_score = resume_score + 7
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added Skills</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Skills. It will help you a lot</h4>''',unsafe_allow_html=True)
                
                # for hobbies 4 points
                if 'HOBBIES' in resume_text:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                elif 'Hobbies' in resume_text:
                    resume_score = resume_score + 4
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Hobbies</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Hobbies. It will show your personality to the Recruiters and give the assurance that you are fit for this role or not.</h4>''',unsafe_allow_html=True)

                # for interests 5 points
                if 'INTERESTS' in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                elif 'Interests' in resume_text:
                    resume_score = resume_score + 5
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Interest</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Interest. It will show your interest other that job.</h4>''',unsafe_allow_html=True)
                
                # for achievements 13 points
                if 'ACHIEVEMENTS' in resume_text:
                    resume_score = resume_score + 13
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                elif 'Achievements' in resume_text:
                    resume_score = resume_score + 13
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Achievements </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Achievements. It will show that you are capable for the required position.</h4>''',unsafe_allow_html=True)

                # for certifications 12 points
                if 'CERTIFICATIONS' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certifications' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                elif 'Certification' in resume_text:
                    resume_score = resume_score + 12
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Certifications </h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Certifications. It will show that you have done some specialization for the required position.</h4>''',unsafe_allow_html=True)

                # for projects 19 points
                if 'PROJECTS' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'PROJECT' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Projects' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                elif 'Project' in resume_text:
                    resume_score = resume_score + 19
                    st.markdown('''<h5 style='text-align: left; color: #1ed760;'>[+] Awesome! You have added your Projects</h4>''',unsafe_allow_html=True)
                else:
                    st.markdown('''<h5 style='text-align: left; color: #000000;'>[-] Please add Projects. It will show that you have done work related the required position or not.</h4>''',unsafe_allow_html=True)

                ### Score Bar
                my_bar = st.progress(0)
                score = 0
                for percent_complete in range(resume_score):
                    score +=1
                    time.sleep(0.1)
                    my_bar.progress(percent_complete + 1)

                st.markdown(f'''
                <div style="padding:15px; border-radius:8px; border-left: 4px solid #5fba7d; margin-bottom:15px;">
                    <h3 style="color:#021659; margin:0 0 5px 0;">Your Resume Score</h3>
                    <p style="font-size:32px; color:#5fba7d; font-weight:bold; margin:5px 0;">{score}/100</p>
                    <p style="color:#666; font-size:14px; margin:5px 0;">Calculated based on resume content</p>
                </div>
                ''', unsafe_allow_html=True)

                st.markdown('''
                <div style="padding:15px; border-radius:8px; border-left: 4px solid #021659; margin:15px 0;">
                    <h3 style="color:#021659; margin:0 0 10px 0;">Relative Score Comparison</h3>
                </div>
                ''', unsafe_allow_html=True)
                
                all_scores = fetch_all_resume_scores()
                average_score = calculate_average_score(all_scores)
                
                # Create columns for comparison
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f'''
                    <div style="background-color:#f8f9fa; padding:12px; border-radius:6px; text-align:center;">
                        <p style="font-weight:bold; margin:5px 0;">Your Score</p>
                        <p style="font-size:24px; color:#5fba7d; font-weight:bold; margin:5px 0;">{score}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f'''
                    <div style="background-color:#f8f9fa; padding:12px; border-radius:6px; text-align:center;">
                        <p style="font-weight:bold; margin:5px 0;">Average Score</p>
                        <p style="font-size:24px; color:#666; font-weight:bold; margin:5px 0;">{average_score:.2f}</p>
                    </div>
                    ''', unsafe_allow_html=True)
                
                # Comparison message
                if score > average_score:
                    st.success("**Your score is above average!** Good job with your resume.")
                elif score == average_score:
                    st.warning("**Your score matches the average.** There's room for improvement.")
                else:
                    st.error("**Below average score.** Consider adding more sections to your resume.")

                # 5. Job Profile Matching Section
                st.subheader("Job Profile Matching")

                # Define keywords for each job profile
                profile_keywords = {
                    'Data Science': ['tensorflow', 'pytorch', 'machine learning', 'deep Learning', 'streamlit', 'python', 
                                    'analytics', 'scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn', 'NLP', 
                                    'computer vision', 'SQL', 'EDA', 'MLOps', 'Reinforcement learning'],
                    'Web Development': ['react', 'django', 'node.js', 'react.js', 'react', 'node', 'php', 'magento', 
                                    'wordpress', 'javascript', 'angular js', 'C#', 'Asp.net', 'flask', 'express.js', 
                                    'mongodb', 'mysql', 'firebase', 'REST API', 'full-stack'],
                    'Android Development': ['android', 'android development', 'flutter', 'kotlin', 'xml', 'kivy'],
                    'iOS Development': ['ios', 'ios development', 'swift', 'cocoa', 'cocoa touch', 'xcode'],
                    'UI/UX Development': ['ux', 'adobe xd', 'figma', 'zeplin', 'balsamiq', 'ui', 'prototyping', 
                                        'wireframes', 'storyframes', 'adobe photoshop', 'photoshop', 'editing', 
                                        'adobe illustrator', 'illustrator', 'adobe after effects', 'after effects', 
                                        'adobe premier pro', 'premier pro', 'adobe indesign', 'indesign', 'wireframe', 
                                        'solid', 'grasp', 'user research', 'user experience']
                }

                # Calculate raw matching counts for each profile
                resume_skills = [skill.lower() for skill in resume_data['skills']]
                raw_counts = {}

                for profile, keywords in profile_keywords.items():
                    matched_skills = [skill for skill in resume_skills if skill in [kw.lower() for kw in keywords]]
                    raw_counts[profile] = len(matched_skills)

                # Calculate normalized percentages that sum to 100%
                total_counts = sum(raw_counts.values())
                if total_counts > 0:
                    normalized_percentages = {profile: (count / total_counts) * 100 for profile, count in raw_counts.items()}
                else:
                    normalized_percentages = {profile: 0 for profile in raw_counts.keys()}

                # Round the percentages and ensure they sum to 100%
                rounded_percentages = {}
                sum_percent = 0

                # First pass - round all percentages
                for profile, percentage in normalized_percentages.items():
                    rounded = round(percentage, 2)
                    rounded_percentages[profile] = rounded
                    sum_percent += rounded

                # Adjust if needed to make sum exactly 100%
                if sum_percent != 100:
                    # Find the profile with largest decimal portion and adjust
                    profile_to_adjust = max(normalized_percentages.items(), 
                                        key=lambda x: x[1] - round(x[1], 2))[0]
                    rounded_percentages[profile_to_adjust] += (100 - sum_percent)

                # Create pie chart with normalized percentages
                fig = px.pie(
                    names=list(rounded_percentages.keys()),
                    values=list(rounded_percentages.values()),
                    title='Job Profile Matching Distribution',
                    color_discrete_sequence=px.colors.sequential.Aggrnyl,
                    hole=0.3
                )
                fig.update_traces(
                    textinfo='percent+label',
                    textposition='inside',
                    insidetextorientation='radial'
                )
                st.plotly_chart(fig)

                # Create table with the same normalized percentages
                matches_df = pd.DataFrame({
                    'Job Profile': list(rounded_percentages.keys()),
                    'Match Percentage': list(rounded_percentages.values())
                }).sort_values('Match Percentage', ascending=False)

                # Reset index to get proper serial numbers (1-5)
                matches_df.reset_index(drop=True, inplace=True)
                matches_df.index = matches_df.index + 1

                # Display the styled version with bars
                st.dataframe(
                    matches_df.style.bar(
                        subset=['Match Percentage'], 
                        color='#5fba7d',
                        vmin=0,
                        vmax=100
                    ).format({'Match Percentage': '{:.2f}%'}),
                    use_container_width=True
                )

                # Show top matching profile
                top_profile = matches_df.iloc[0]['Job Profile']
                top_percentage = matches_df.iloc[0]['Match Percentage']
                st.success(f"Your resume matches best with: {top_profile} ({top_percentage:.2f}%)")

                # 6. Skill Gap Analysis Section
                st.subheader("Skill Gap Analysis")

                if 'skills' in resume_data and resume_data['skills']:
                    gap_analysis = skill_gap_analyzer.analyze_gaps(
                        resume_data['skills'], 
                        top_profile
                    )
                    
                    if gap_analysis:
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=gap_analysis['match_percentage'],
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Skill Match Percentage"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "#5fba7d"},
                                'steps': [
                                    {'range': [0, 50], 'color': "#cbcbcb"},
                                    {'range': [50, 75], 'color': "#a4a4a4"},
                                    {'range': [75, 100], 'color': "#7a7a7a"}
                                ]
                            }
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.success("**Your Top 5 Relevant Skills:**")
                            if gap_analysis['your_skills']:
                                # Get top 5 most relevant skills (prioritize skills that appear first in required_skills list)
                                required_order = {skill: idx for idx, skill in enumerate(gap_analysis['all_required_skills'])}
                                top_user_skills = sorted(
                                    gap_analysis['your_skills'],
                                    key=lambda x: required_order.get(x.lower(), len(gap_analysis['all_required_skills'])))
                                top_user_skills = top_user_skills[:5]

                                for skill in top_user_skills:
                                    st.markdown(f"""
                                    <div style="background-color:#e6f7e6; padding:10px; border-radius:5px; 
                                                margin:5px 0; border-left:5px solid #5fba7d">
                                        <span style="font-weight:bold">✓ {skill.capitalize()}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.warning("No relevant skills found for this job profile")
                        
                        with col2:
                            st.warning("**Top 5 Skills to Develop:**")
                            if gap_analysis['missing_skills']:
                                # Get top 5 most important missing skills (prioritize skills that appear first in required_skills list)
                                required_order = {skill: idx for idx, skill in enumerate(gap_analysis['all_required_skills'])}
                                top_missing_skills = sorted(
                                    gap_analysis['missing_skills'],
                                    key=lambda x: required_order.get(x.lower(), len(gap_analysis['all_required_skills'])))
                                top_missing_skills = top_missing_skills[:5]
                                
                                for skill in top_missing_skills:
                                    st.markdown(f"""
                                    <div style="background-color:#fff8e6; padding:10px; border-radius:5px; 
                                                margin:5px 0; border-left:5px solid #ffcc00">
                                        <span style="font-weight:bold">+ {skill.capitalize()}</span>
                                    </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.success("Great job! You have all key skills for this profile")
                        
                        # Add radar chart to visualize skill match
                        st.subheader("Skill Radar Chart")
                        
                        # Prepare data for radar chart
                        categories = gap_analysis['top_skills'][:5]
                        required_levels = [5] * len(categories)
                        
                        # Calculate user's skill levels (1 if present, 0 if missing)
                        user_levels = []
                        for skill in categories:
                            if skill.lower() in [s.lower() for s in resume_data['skills']]:
                                user_levels.append(5)  # Present skill gets max level
                            else:
                                user_levels.append(1)  # Missing skill gets min level
                        
                        # Create radar chart
                        fig = go.Figure()
                        
                        fig.add_trace(go.Scatterpolar(
                            r=required_levels,
                            theta=categories,
                            fill='toself',
                            name='Required Skills',
                            line_color='lightgray'
                        ))
                        
                        fig.add_trace(go.Scatterpolar(
                            r=user_levels,
                            theta=categories,
                            fill='toself',
                            name='Your Skills',
                            line_color='#5fba7d'
                        ))
                        
                        fig.update_layout(
                            polar=dict(
                                radialaxis=dict(
                                    visible=True,
                                    range=[0, 5]
                                )),
                            showlegend=True,
                            title="Your Skills vs Required Skills"
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)

                st.header("Recommended Videos for You")
                
                st.subheader("Resume Writing Tips")
                resume_vid = random.choice(resume_videos)
                st.video(resume_vid)

                st.subheader("Interview Preparation Tips")
                interview_vid = random.choice(interview_videos)
                st.video(interview_vid)

                if top_profile == 'Data Science':
                    rec_course = course_recommender(ds_course)
                elif top_profile == 'Web Development':
                    rec_course = course_recommender(webdev_course)
                elif top_profile == 'Android Development':
                    rec_course = course_recommender(android_course)
                elif top_profile == 'iOS Development':
                    rec_course = course_recommender(ios_course)
                elif top_profile == 'UI/UX Development':
                    rec_course = course_recommender(uiux_course)

                ### Getting Current Date and Time
                ts = time.time()
                cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                timestamp = str(cur_date+'_'+cur_time)

                ## Calling insert_data to add all the data into user_data                
                insert_data(str(sec_token), str(ip_add), (host_name), (dev_user), (os_name_ver), (latlong), (city), (state), (country), (act_name), (act_mail), (act_mob), resume_data['name'], resume_data['email'], str(resume_score), timestamp, str(resume_data['no_of_pages']), str(resume_data['skills']), pdf_name)

            else:
                st.error('Something went wrong..')               


    ###### CODE FOR FEEDBACK SIDE ######
    elif choice == 'Feedback':   
    # timestamp 
        ts = time.time()
        cur_date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
        cur_time = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
        timestamp = str(cur_date+'_'+cur_time)

        with st.form("my_form", clear_on_submit=True):
            st.subheader("Feedback Form")
            feed_name = st.text_input('Name*', key='feed_name')
            feed_email = st.text_input('Email*', key='feed_email')
            feed_score = st.slider('Rate Us From 1 - 5*', 1, 5, key='feed_score')
            comments = st.text_area('Comments', key='comments')
            submitted = st.form_submit_button("Submit Feedback")
            
        if submitted:

            if not feed_name or not feed_email or not feed_score:
                st.error("Please fill all required fields (marked with *)")
            else:
                insertf_data(feed_name, feed_email, feed_score, comments, timestamp)    
                st.success("Thank you! Your feedback has been recorded.")

    
    ###### CODE FOR ABOUT PAGE ######
    elif choice == 'About':   
        st.subheader("About The AI Resume Analyzer")

        st.markdown('''
        <h4>How It Works</h4>
        <p align='justify'>
        This tool uses AI to analyze resumes and provide career recommendations:
        </p>
        <ol>
        <li><b>Upload:</b> Submit your resume in PDF format</li>
        <li><b>Analysis:</b> Our system extracts skills, experience, and education</li>
        <li><b>Matching:</b> Compares your profile against 5 career paths</li>
        <li><b>Scoring:</b> Evaluates resume completeness and effectiveness</li>
        <li><b>Recommendations:</b> Suggests skills to improve and relevant courses</li>
        </ol>

        <h4>Key Features</h4>
        <ul>
        <li>Skill gap analysis with visual charts</li>
        <li>Resume score compared to industry averages</li>
        <li>Personalized course recommendations</li>
        <li>Interview preparation resources</li>
        <li>Secure and private processing</li>
        </ul>

        <h4>For Best Results</h4>
        <p align='justify'>
        - Use a well-structured resume<br>
        - Review all analysis sections<br>
        - Focus on recommended skill improvements<br>
        - Check suggested courses and videos
        </p>
        ''', unsafe_allow_html=True)

    ###### CODE FOR ADMIN PAGE ######
    elif choice == 'Admin':
        st.success('Welcome to Admin Side')

        ad_user = st.text_input("Username")
        ad_password = st.text_input("Password", type='password')

        if st.button('Login'):
            if ad_user == 'admin' and ad_password == '123':
                all_scores = fetch_all_resume_scores()
                average_score = calculate_average_score(all_scores)

                st.subheader("Average Resume Score")
                st.write(f"The average resume score across all users is: **{average_score:.2f}**")

                ### Fetch miscellaneous data from user_data(table) and convert it into dataframe
                cursor.execute('''SELECT ID, ip_add, resume_score, city, state, country from user_data''')
                datanalys = cursor.fetchall()
                plot_data = pd.DataFrame(datanalys, columns=['ID', 'IP_add', 'resume_score', 'City', 'State', 'Country'])
                
                values = plot_data.ID.count()
                st.success(f"Welcome Admin! Total {values} Users Have Used Our Tool :)")                
                
                cursor.execute('''SELECT ID, act_name as Name, act_mail as Email, act_mob as Mobile, 
                                resume_score, Page_no as Total_Pages, pdf_name as File_Name, 
                                city as City, state as State, country as Country 
                                from user_data''')
                data = cursor.fetchall()                

                df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Mobile', 
                                                'Resume Score', 'Total Pages', 
                                                'File Name', 'City', 'State', 'Country'])
                

                df['Resume Score'] = pd.to_numeric(df['Resume Score'], errors='coerce')

                df['Relative Score'] = df['Resume Score'].apply(lambda x: 'Above Average' if x > average_score else ('Equal to Average' if x == average_score else 'Below Average'))

                st.header("User's Data with Relative Scoring")
                st.dataframe(df.set_index('ID'), use_container_width=True)

                relative_score_counts = df['Relative Score'].value_counts()

                st.markdown(get_csv_download_link(df,'User_Data.csv','Download Report'), unsafe_allow_html=True)

                cursor.execute('''SELECT * from user_feedback''')
                data = cursor.fetchall()

                st.header("User's Feedback Data")
                feedback_df = pd.DataFrame(data, columns=['ID', 'Name', 'Email', 'Feedback Score', 'Comments', 'Timestamp'])
                st.dataframe(feedback_df.set_index('ID'), use_container_width=True)

                st.markdown(get_csv_download_link(feedback_df,'User_Feedback_Data.csv','Download Feedback Data'), unsafe_allow_html=True)
                
                ### Analyzing All the Data's in pie charts
                st.subheader("Distribution of Relative Scores")
                fig = px.pie(values=relative_score_counts, 
                            names=relative_score_counts.index, 
                            title="Percentage of Users with Above, Equal to, and Below Average Scores",
                            color_discrete_sequence=px.colors.sequential.Agsunset)
                st.plotly_chart(fig)

                # Fetching feedback scores
                query = 'select * from user_feedback'
                plotfeed_data = pd.read_sql(query, connection)   
                labels = plotfeed_data.feed_score.unique()
                values = plotfeed_data.feed_score.value_counts()
                
                # Pie chart for user ratings
                st.subheader("User Ratings")
                fig = px.pie(values=values, 
                            names=labels, 
                            title="User Rating Distribution (1-5)",
                            color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig)

                # Resume score distribution
                labels = plot_data.resume_score.unique()                
                values = plot_data.resume_score.value_counts()

                # Pie chart for Resume Score
                st.subheader("Resume Score Distribution")
                fig = px.pie(plot_data, 
                            values=values, 
                            names=labels, 
                            title='Resume Scores (1-100)',
                            color_discrete_sequence=px.colors.sequential.Sunsetdark)
                st.plotly_chart(fig)
            else:
                st.error("Wrong ID & Password Provided")
skill_gap_analyzer = SkillGapAnalyzer()
run()