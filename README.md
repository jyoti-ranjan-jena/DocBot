# Sample Python/Streamlit

DocBot is a Streamlit web application that enables interactive conversations with PDF documents using Gemini AI, providing insightful responses based on the document's content.

---

## 🚀 Getting Started  

### Open Using Daytona  

1. **Install Daytona**: Follow the [Daytona installation guide](https://www.daytona.io/docs/installation/installation/).

2. **Create the Workspace**:  
   ```bash  
   daytona create https://github.com/jyoti-ranjan-jena/DocBot.git 
   ```

3. **Add gemini api key**:
   Add this in .env file
   ```bash  
   GOOGLE_API_KEY=<API_KEY>
   ```

4. **Start the Application**:  
   ```bash  
   streamlit run app.py
   ```

---

## ✨ Features  

- PDF Upload: Easily upload PDF documents to interact with.
- Interactive Chat: Engage in real-time conversations with the content of your PDFs using Gemini AI.
- Insightful Responses: Receive detailed and contextually accurate answers based on the document's content.
- User-Friendly Interface: Enjoy a seamless experience with a clean and intuitive Streamlit interface.
- Secure API Integration: Utilize your Google API key securely for AI interactions.
