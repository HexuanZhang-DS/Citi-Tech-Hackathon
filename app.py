from flask import Flask, render_template, redirect, url_for, request
app = Flask(__name__)

@app.route("/")
def hello():
  print("Handling request to home page.")
  return render_template('student_homepage.html')

@app.route('/investor_landing',methods = ['GET'])
def investor_landing():
   if request.method == 'GET':
      return render_template('Investor_logged.html')
   else:
      return render_template('student_homepage.html')

@app.route('/ISA_form',methods = ['GET'])
def ISA_form():
   if request.method == 'GET':
      return render_template('ISA_form.html')
   else:
      return render_template('student_homepage.html')

if __name__ == "__main__":
  app.run()
