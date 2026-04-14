"""
app.py - Flask web application for MicroScan: Microplastic Detection Platform
All image data handled as base64 strings. Nothing saved to disk.
"""

import datetime
import uuid

from flask import (
    Flask, render_template, request, redirect,
    url_for, session, jsonify
)

from processing import (
    file_to_b64, run_pipeline, img_to_b64, b64_to_img,
    draw_detections, detect_contours
)
from utils import (
    generate_single_pdf,
    make_thumbnail, generate_single_chart
)

app = Flask(__name__)
app.secret_key = "microscan_secret_key_2025"

# In-memory session store keyed by session ID
# { session_id: [ history_entry, ... ] }
_history_store = {}


def get_history():
    sid = session.get('sid')
    if not sid:
        return []
    return _history_store.get(sid, [])


def push_history(entry):
    sid = session.setdefault('sid', str(uuid.uuid4()))
    _history_store.setdefault(sid, []).insert(0, entry)  # newest first


# ─────────────────────────────────────────────
# HOME
# ─────────────────────────────────────────────

@app.route('/')
def index():
    history = get_history()
    recent  = history[:5]  # sidebar preview
    return render_template('index.html', recent=recent, history_count=len(history))


# ─────────────────────────────────────────────
# SINGLE IMAGE
# ─────────────────────────────────────────────

@app.route('/single', methods=['GET', 'POST'])
def single():
    if request.method == 'GET':
        return render_template('single.html')

    file = request.files.get('image')
    if not file or file.filename == '':
        return render_template('single.html', error="Please upload an image file.")

    try:
        original_b64 = file_to_b64(file)
        filename     = file.filename
        result       = run_pipeline(original_b64)
        chart_b64    = generate_single_chart(result['class_counts'])
        timestamp    = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Generate PDF (base64, not saved)
        pdf_b64 = generate_single_pdf(filename, result, result['steps']['cropped_roi'], timestamp)

        # Build history entry
        thumb = make_thumbnail(result['steps']['cropped_roi'])
        entry = {
            'id':        str(uuid.uuid4()),
            'filename':  filename,
            'timestamp': timestamp,
            'count':     result['count'],
            'score':     result['score'],
            'level':     result['level'],
            'thumbnail': thumb,
            'result':    result,        # full result for "view again"
            'original':  result['steps']['cropped_roi'],
            'pdf_b64':   pdf_b64,
        }
        push_history(entry)

        return render_template(
            'single.html',
            result=result,
            filename=filename,
            timestamp=timestamp,
            chart_b64=chart_b64,
            pdf_b64=pdf_b64,
        )

    except Exception as e:
        return render_template('single.html', error=f"Processing error: {str(e)}")


# ─────────────────────────────────────────────
# SINGLE STEPS VIEW
# ─────────────────────────────────────────────

@app.route('/single/steps', methods=['POST'])
def single_steps():
    """Receive POSTed result data and show step-by-step images."""
    # Steps are passed as hidden base64 fields from the form
    steps_keys = ['cropped_roi', 'enhancement', 'histogram',
                  'noise_removal', 'spatial', 'threshold',
                  'morphology', 'final']
    steps = {}
    for k in steps_keys:
        val = request.form.get(f'step_{k}')
        if val:
            steps[k] = val

    filename  = request.form.get('filename', 'image')
    count     = request.form.get('count', '0')
    score     = request.form.get('score', '0')
    level     = request.form.get('level', 'Low')

    return render_template(
        'single_steps.html',
        steps=steps,
        filename=filename,
        count=count,
        score=score,
        level=level,
    )



# ─────────────────────────────────────────────
# DASHBOARD
# ─────────────────────────────────────────────

@app.route('/dashboard')
def dashboard():
    history = get_history()
    return render_template('dashboard.html', history=history)


@app.route('/dashboard/view/<entry_id>')
def view_result(entry_id):
    """Re-display a past result from session history."""
    history = get_history()
    entry   = next((e for e in history if e['id'] == entry_id), None)
    if not entry:
        return redirect(url_for('dashboard'))

    result    = entry['result']
    chart_b64 = generate_single_chart(result['class_counts'])

    # Regenerate PDF if not cached
    pdf_b64 = entry.get('pdf_b64')
    if not pdf_b64:
        pdf_b64 = generate_single_pdf(
            entry['filename'], result, entry['original'], entry['timestamp'])
        entry['pdf_b64'] = pdf_b64

    return render_template(
        'single.html',
        result=result,
        filename=entry['filename'],
        timestamp=entry['timestamp'],
        chart_b64=chart_b64,
        pdf_b64=pdf_b64,
        from_history=True,
    )


@app.route('/dashboard/clear', methods=['POST'])
def clear_history():
    sid = session.get('sid')
    if sid and sid in _history_store:
        _history_store[sid] = []
    return redirect(url_for('dashboard'))


# ─────────────────────────────────────────────
# API: download PDF (base64 → browser download)
# ─────────────────────────────────────────────

@app.route('/download/pdf/<entry_id>')
def download_pdf(entry_id):
    """Stream PDF download for a history entry."""
    import base64
    from flask import make_response

    history = get_history()
    entry   = next((e for e in history if e['id'] == entry_id), None)
    if not entry:
        return "Not found", 404

    pdf_b64 = entry.get('pdf_b64')
    if not pdf_b64:
        pdf_b64 = generate_single_pdf(
            entry['filename'], entry['result'], entry['original'], entry['timestamp'])
        entry['pdf_b64'] = pdf_b64

    pdf_bytes = base64.b64decode(pdf_b64)
    resp = make_response(pdf_bytes)
    resp.headers['Content-Type'] = 'application/pdf'
    safe_name = entry['filename'].replace(' ', '_').rsplit('.', 1)[0]
    resp.headers['Content-Disposition'] = f'attachment; filename=microscan_{safe_name}.pdf'
    return resp


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
