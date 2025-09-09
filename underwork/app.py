"""
ML Launch Pad — Title-Based Project Scoping Tool (Flask)

This single-file app + templates implements the flow you requested:
- User supplies a *project title* (e.g. "Cat vs Dog Classifier").
- App asks Gemini to identify the ML task and keywords.
- Uses Hugging Face Hub to find candidate models for that task.
- Calls Gemini again to produce an end-to-end project plan, estimated model accuracies, step-by-step approach, tools & packages, deployment suggestions, and next steps.
- Attempts to download a suitable pre-trained TFLite model (if available in the map).
- Generates a visualization (estimated model accuracy bar chart) and a PDF report with all results.

Files created by this single-file app (placed in /outputs):
- report_<id>.pdf  — final project PDF
- model_<id>.tflite — downloaded TFLite model (if available)
- viz_<id>.png — performance visualization (if Gemini provides estimations in expected format)

Templates (placed in templates/):
- index.html — project title input page
- results.html — human-friendly results page with quick download buttons

IMPORTANT NOTES BEFORE RUNNING
- Set environment variables: HF_TOKEN (Hugging Face token) and GEMINI_API_KEY (Google Gen AI key).
- The script uses the `huggingface_hub` HfApi and `google.generativeai` (genai). Install dependencies:

pip install flask huggingface_hub google-generativeai reportlab matplotlib requests markdown

(If you want local TFLite model support, also install `tflite-runtime` or `tensorflow`.)

USAGE
1. Save this file as `ml_launchpad_title.py`.
2. Create a `templates/` folder and the two templates (index.html, results.html) — the app will create them automatically if you prefer.
3. Run: `python ml_launchpad_title.py` and open http://127.0.0.1:5000/

Security & Quotas
- Calling Gemini and Hugging Face APIs will use your API keys and may incur usage costs. Use prudently and cache results where possible.
- The app downloads (and stores) TFLite models into the `outputs/` folder — double-check model URLs before distributing.

"""

from flask import Flask, request, render_template_string, send_file, redirect, url_for
import os
import time
import json
import re
import requests
import matplotlib.pyplot as plt
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import letter
import markdown

from huggingface_hub import HfApi
import google.generativeai as genai

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get('HF_TOKEN', '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') 
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', 'AIzaSyC6dtvjEzKrLzOYXKkrbV6EUx6OkbfPbiY')

# User-editable map of task -> TFLite model URL (optional)
TFLITE_MODEL_MAP = {
    'Image Classification': 'https://tfhub.dev/tensorflow/lite-model/mobilenet_v2_1.0_224/1?lite-format=tflite',
    'Text Classification': 'https://tfhub.dev/tensorflow/lite-model/mobilebert/1?lite-format=tflite',
}

app = Flask(__name__)
app.config['OUTPUT_FOLDER'] = 'outputs'
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

hf_api = HfApi()
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ---------- UTILITIES ----------

def call_gemini_for_task_id(project_title):
    """Ask Gemini to return a compact JSON with primary_task and search_keywords."""
    if not GEMINI_API_KEY:
        return {"primary_task": "Unknown", "search_keywords": [project_title]}
    prompt = f"""
Analyze the following project title and determine its primary machine learning task.
Also provide 3 relevant search keywords as a JSON object.

Project Title: "{project_title}"

Return strictly valid JSON like:
{{"primary_task": "Image Classification", "search_keywords": ["image classification","birds","vision"]}}
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        resp = model.generate_content(prompt)
        txt = resp.text.strip()
        # try to extract JSON from the response
        json_text = re.sub(r"^.*?\{","{", txt, count=1)
        data = json.loads(json_text)
        return data
    except Exception as e:
        print('Gemini task id error:', e)
        return {"primary_task": "Unknown", "search_keywords": [project_title]}


def search_hf_models(task, keywords, limit=5):
    q = f"{task} " + " ".join(keywords)
    try:
        models = hf_api.list_models(search=q, limit=limit)
        return [m.modelId for m in models]
    except Exception as e:
        print('HF search error:', e)
        return []


def call_gemini_for_analysis(project_title, task, models):
    if not GEMINI_API_KEY:
        return 'Gemini not configured.'
    models_text = '\n'.join([f"- {m}" for m in models]) or 'None'
    prompt = f"""
You are an expert ML project manager. Given the project title and candidate models, produce a detailed project plan.

Project Title: {project_title}
Identified Task: {task}
Candidate Models:\n{models_text}

Produce markdown sections: ### Project Overview, ### Performance Estimation, ### Project Roadmap, ### Tools & Packages, ### Suggested Dataset Sources, ### Deployment Recommendation
"""
    try:
        model = genai.GenerativeModel('gemini-1.5-mini')
        resp = model.generate_content(prompt)
        return resp.text
    except Exception as e:
        print('Gemini analysis error:', e)
        return f'Gemini error: {e}'


def generate_tflite_model(task, out_path):
    url = TFLITE_MODEL_MAP.get(task)
    if not url:
        return None
    try:
        r = requests.get(url, allow_redirects=True, timeout=30)
        r.raise_for_status()
        with open(out_path, 'wb') as f:
            f.write(r.content)
        return out_path
    except Exception as e:
        print('Failed to download TFLite model:', e)
        return None


def create_visualization(gemini_text, out_path):
    try:
        # look for patterns like "ModelName: 80-90% accuracy"
        estimates = re.findall(r"([\w\s/-]+):\s*(\d+)-(\d+)%\s*accuracy", gemini_text, flags=re.IGNORECASE)
        if not estimates:
            return None
        models = [e[0].strip() for e in estimates]
        lowers = [int(e[1]) for e in estimates]
        uppers = [int(e[2]) for e in estimates]
        avgs = [(l+u)/2 for l,u in zip(lowers, uppers)]
        erros = [(u-l)/2 for l,u in zip(lowers, uppers)]
        plt.figure(figsize=(8,4))
        plt.bar(models, avgs, yerr=erros, capsize=5)
        plt.ylabel('Estimated Accuracy (%)')
        plt.ylim(0,100)
        plt.xticks(rotation=20, ha='right')
        plt.tight_layout()
        plt.savefig(out_path)
        plt.close()
        return out_path
    except Exception as e:
        print('Visualization error:', e)
        return None


def create_pdf_report(report_path, context):
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    story = []
    story.append(Paragraph('<b>ML Launch Pad - Project Scoping Report</b>', styles['Title']))
    story.append(Spacer(1,12))
    story.append(Paragraph(f"<b>Project:</b> {context.get('project_title')}", styles['Heading2']))
    story.append(Paragraph(f"<b>Identified Task:</b> {context.get('task')}", styles['Normal']))
    story.append(Spacer(1,10))
    story.append(Paragraph('<b>Suggested Hugging Face Models</b>', styles['Heading2']))
    for m in context.get('hf_models',[]):
        story.append(Paragraph(f"- {m}", styles['Normal']))
    story.append(Spacer(1,10))
    story.append(Paragraph('<b>Gemini Analysis</b>', styles['Heading2']))
    story.append(Paragraph(context.get('gemini_analysis','').replace('\n','<br/>'), styles['Normal']))
    if context.get('viz_path'):
        story.append(Spacer(1,12))
        story.append(Paragraph('<b>Estimated Performance Comparison</b>', styles['Heading2']))
        story.append(RLImage(context['viz_path'], width=450, height=250))
    doc.build(story)
    return report_path
def search_hf_datasets(task, keywords, limit=3):
    """Search Hugging Face for relevant datasets."""
    q = f"{task} " + " ".join(keywords)
    try:
        datasets = hf_api.list_datasets(search=q, limit=limit)
        # Return dataset name and directory (repo_id)
        return [{"name": d.id, "dir": d.id} for d in datasets]
    except Exception as e:
        print('HF dataset search error:', e)
        return []

# ---------- Flask HTML templates ----------
INDEX_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Launch Pad</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container py-5">
  <h1 class="mb-4">ML Launch Pad — Project Scoping</h1>
  <form method="post" action="/generate" class="row g-3">
    <div class="col-12">
      <input name="project_title" class="form-control form-control-lg" placeholder="e.g. Detect ripe mangoes in orchard" required>
    </div>
    <div class="col-auto">
      <button class="btn btn-primary btn-lg" type="submit">Generate Plan</button>
    </div>
  </form>
  <p class="mt-3 text-muted">This will call Gemini & Hugging Face. Make sure your API keys are set.</p>
</body>
</html>
'''

RESULTS_HTML = '''
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ML Plan Ready</title>
  <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
</head>
<body class="container py-5">
  <h1 class="mb-3">Your ML Plan is Ready ✅</h1>
  <p><strong>Project:</strong> {{ project_title }}</p>
  <p><strong>Identified Task:</strong> {{ task }}</p>
  <hr>
  <h4>Suggested Hugging Face Models</h4>
  <ul>
    {% for m in hf_models %}
      <li>{{ m }}</li>
    {% endfor %}
  </ul>
  {% if viz_exists %}
    <h4>Estimated Performance</h4>
    <img src="/{{ viz_path_url }}" class="img-fluid" alt="viz">
  {% endif %}
  <h4 class="mt-3">Gemini Analysis (summary)</h4>
  <div class="card p-3">
    <div>{{ gemini_analysis_html|safe }}</div>
  </div>
  <div class="mt-4">
    <a class="btn btn-success" href="{{ url_for('download_file', name=report_filename) }}">Download PDF Report</a>
    {% if tflite_filename %}
      <a class="btn btn-secondary ms-2" href="{{ url_for('download_file', name=tflite_filename) }}">Download TFLite Model</a>
    {% endif %}
  </div>
  <p class="mt-4"><a href="/">Start another</a></p>
</body>
</html>
'''


# ---------- Flask routes ----------
@app.route('/')
def index():
    return render_template_string(INDEX_HTML)


@app.route('/generate', methods=['POST'])
def generate():
    project_title = request.form.get('project_title')
    if not project_title:
        return 'Missing project title', 400
    job_id = str(int(time.time()))
    context = {'project_title': project_title}

    # 1) Identify task via Gemini
    task_info = call_gemini_for_task_id(project_title)
    task = task_info.get('primary_task', 'Unknown')
    context['task'] = task
    context['search_keywords'] = task_info.get('search_keywords', [])

    # 2) Search Hugging Face Models
    hf_models = search_hf_models(task, context['search_keywords'], limit=5)
    context['hf_models'] = hf_models

    # 2b) Search Hugging Face Datasets
    hf_datasets = search_hf_datasets(task, context['search_keywords'], limit=3)
    context['hf_datasets'] = hf_datasets

    # 3) Gemini in-depth analysis
    gemini_analysis = call_gemini_for_analysis(project_title, task, hf_models)
    context['gemini_analysis'] = gemini_analysis

    # 4) Generate visualization (if possible)
    gemini_html = markdown.markdown(gemini_analysis)
    viz_file = None
    try:
        viz_file = os.path.join(app.config['OUTPUT_FOLDER'], f'viz_{job_id}.png')
        viz_file = create_visualization(gemini_analysis, viz_file)
    except Exception as e:
        viz_file = None

    # 5) Download TFLite model (if available)
    tflite_name = None
    tflite_path = None
    if task in TFLITE_MODEL_MAP:
        tflite_name = f"model_{job_id}.tflite"
        tflite_path = os.path.join(app.config['OUTPUT_FOLDER'], tflite_name)
        generate_tflite_model(task, tflite_path)

    # 6) Generate PDF report
    report_path = os.path.join(app.config['OUTPUT_FOLDER'], f'report_{job_id}.pdf')
    context['viz_path'] = viz_file
    create_pdf_report(report_path, context)

    return render_template_string(
        RESULTS_HTML,
        project_title=project_title,
        task=task,
        hf_models=hf_models,
        hf_datasets=hf_datasets,  # Pass datasets to template
        gemini_analysis_html=gemini_html,
        viz_exists=bool(viz_file),
        viz_path_url=os.path.join(app.config['OUTPUT_FOLDER'], os.path.basename(viz_file)) if viz_file else '',
        report_filename=os.path.basename(report_path),
        tflite_filename=tflite_name
    )

@app.route('/outputs/<path:filename>')
def static_outputs(filename):
    return send_file(os.path.join(app.config['OUTPUT_FOLDER'], filename))


@app.route('/download/<name>')
def download_file(name):
    path = os.path.join(app.config['OUTPUT_FOLDER'], name)
    if not os.path.exists(path):
        return 'File not found', 404
    return send_file(path, as_attachment=True)


if __name__ == '__main__':
    app.run(debug=True)
