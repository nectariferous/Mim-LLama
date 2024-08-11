from flask import Flask, render_template, jsonify
import json

app = Flask(__name__)

@app.route('/')
def index():
    try:
        with open('output.json', 'r') as file:
            apps = json.load(file)
    except IOError as e:
        return f"Error reading file: {e}", 500
    except json.JSONDecodeError as e:
        return f"Error decoding JSON: {e}", 500

    return render_template('index.html', apps=apps)

@app.route('/api/apps')
def api_apps():
    try:
        with open('output.json', 'r') as file:
            apps = json.load(file)
    except IOError as e:
        return jsonify({"error": f"Error reading file: {e}"}), 500
    except json.JSONDecodeError as e:
        return jsonify({"error": f"Error decoding JSON: {e}"}), 500

    return jsonify(apps)

if __name__ == '__main__':
    app.run(debug=True)
