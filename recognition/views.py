from django.shortcuts import render,redirect
from .forms import usernameForm,DateForm,UsernameAndDateForm, DateForm_2,Holiday_form
from django.contrib import messages
from django.contrib.auth.models import User
import cv2
import dlib
import imutils
from imutils import face_utils
from imutils.video import VideoStream
from imutils.face_utils import rect_to_bb
from imutils.face_utils import FaceAligner
import time
from attendance_system_facial_recognition.settings import BASE_DIR
import os
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
#import sklearn.svm._classes
import sklearn
import numpy as np
from django.contrib.auth.decorators import login_required
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import datetime
from django_pandas.io import read_frame
from users.models import Present, Time ,Holiday
import seaborn as sns
import pandas as pd
from django.db.models import Count
import mpld3
import matplotlib.pyplot as plt
from pandas.plotting import register_matplotlib_converters
from matplotlib import rcParams
import math
import time
from datetime import timedelta , date
from wsgiref.util import FileWrapper
from django.http import StreamingHttpResponse
from PIL import Image


mpl.use('Agg')



#utility functions:
def username_present(username):
	if User.objects.filter(username=username).exists():
		return True
	
	return False

def create_dataset(username):
	id = username
	if(os.path.exists('face_recognition_data/training_dataset/{}/'.format(id))==False):
		os.makedirs('face_recognition_data/training_dataset/{}/'.format(id))
	directory='face_recognition_data/training_dataset/{}/'.format(id)

	#Detect face
	#Loading the HOG face detector and the shape predictpr for allignment

	print("[INFO] Loading the facial detector")
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   #Add path to the shape predictor ######CHANGE TO RELATIVE PATH LATER
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	#capture images from the webcam and process and detect the face
	# Initialize the video stream
	print("[INFO] Initializing Video stream")
	vs = VideoStream(src=0).start()
	#time.sleep(2.0) ####CHECK######

	# Our identifier
	# We will put the id here and we will store the id with a face, so that later we can identify whose face it is
	
	# Our dataset naming counter
	sampleNum = 0
	# Capturing the faces one by one and detect the faces and showing it on the window
	while(True):
		# Capturing the image
		#vs.read each frame
		frame = vs.read()
		#Resize each image
		frame = imutils.resize(frame ,width = 800)
		#the returned img is a colored image but for the classifier to work we need a greyscale image
		#to convert
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#To store the faces
		#This will detect all the images in the current frame, and it will return the coordinates of the faces
		#Takes in image and some other parameter for accurate result
		faces = detector(gray_frame,0)
		#In above 'faces' variable there can be multiple faces so we have to get each and every face and draw a rectangle around it.
		
		
			


		for face in faces:
			print("inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			sampleNum = sampleNum+1
			
			if face is None:
				print("face is none")
				continue

			cv2.imwrite(directory+'/'+str(sampleNum)+'.jpg'	, face_aligned)
			face_aligned = imutils.resize(face_aligned ,width = 400)
			#cv2.imshow("Image Captured",face_aligned)
			# @params the initial point of the rectangle will be x,y and
			# @params end point will be x+width and y+height
			# @params along with color of the rectangle
			# @params thickness of the rectangle
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Add Images",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		cv2.waitKey(1)
		#To get out of the loop
		if(sampleNum>10):
			break
	
	#Stoping the videostream
	vs.stop()
	# destroying all the windows
	cv2.destroyAllWindows()


def predict(face_aligned,svc,threshold=0.7):
	face_encodings=np.zeros((1,128))
	try:
		x_face_locations=face_recognition.face_locations(face_aligned)
		faces_encodings=face_recognition.face_encodings(face_aligned,known_face_locations=x_face_locations)
		if(len(faces_encodings)==0):
			return ([-1],[0])

	except:

		return ([-1],[0])

	prob=svc.predict_proba(faces_encodings)
	result=np.where(prob[0]==np.amax(prob[0]))
	if(prob[0][result[0]]<=threshold):
		return ([-1],prob[0][result[0]])

	return (result[0],prob[0][result[0]])


def vizualize_Data(embedded, targets,):
	
	X_embedded = TSNE(n_components=2).fit_transform(embedded)

	for i, t in enumerate(set(targets)):
		idx = targets == t
		plt.scatter(X_embedded[idx, 0], X_embedded[idx, 1], label=t)

	plt.legend(bbox_to_anchor=(1, 1));
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()	
	plt.savefig('./recognition/static/recognition/img/training_visualisation.png')
	plt.close()



def update_attendance_in_db_in(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		try:
		   qs=Present.objects.get(user=user,date=today)
		except :
			qs= None
		
		if qs is None:
			if present[person]==True:
						a=Present(user=user,date=today,present=True)
						a.save()
			#else:
				#a=Present(user=user,date=today,present=False)
				#a.save()
		else:
			if present[person]==True:
				qs.present=True
				qs.save(update_fields=['present'])
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=False)
			a.save()


def update_attendance_in_db_out(present):
	today=datetime.date.today()
	time=datetime.datetime.now()
	for person in present:
		user=User.objects.get(username=person)
		if present[person]==True:
			a=Time(user=user,date=today,time=time, out=True)
			a.save()
		




def check_validity_times(times_all):

	if(len(times_all)>0):
		sign=times_all.first().out
	else:
		sign=True
	times_in=times_all.filter(out=False)
	times_out=times_all.filter(out=True)
	if(len(times_in)!=len(times_out)):
		sign=True
	break_hourss=0
	if(sign==True):
			check=False
			break_hourss=0
			return (check,break_hourss)
	prev=True
	prev_time=times_all.first().time

	for obj in times_all:
		curr=obj.out
		if(curr==prev):
			check=False
			break_hourss=0
			return (check,break_hourss)
		if(curr==False):
			curr_time=obj.time
			to=curr_time
			ti=prev_time
			break_time=((to-ti).total_seconds())/3600
			break_hourss+=break_time


		else:
			prev_time=obj.time

		prev=curr

	return (True,break_hourss)


def convert_hours_to_hours_mins(hours):
	
	h=int(hours)
	hours-=h
	m=hours*60
	m=math.ceil(m)
	return str(str(h)+ " hrs " + str(m) + "  mins")

		

#used
def hours_vs_date_given_employee(present_qs,time_qs,admin=True):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	qs=present_qs

	for obj in qs:
	
		date=obj.date
		times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		times_all=time_qs.filter(date=date).order_by('time')
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		
		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss
		else:
			obj.break_hours=0

		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
			
	df = read_frame(qs)	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours
	
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	

#used
def hours_vs_employee_given_date(present_qs,time_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]
	df_username=[]
	qs=present_qs

	for obj in qs:
		user=obj.user
		times_in=time_qs.filter(user=user).filter(out=False)
		times_out=time_qs.filter(user=user).filter(out=True)
		times_all=time_qs.filter(user=user)
		obj.time_in=None
		obj.time_out=None
		obj.hours=0
		obj.hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
		if (len(times_out)>0):
			obj.time_out=times_out.last().time
		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		else:
			obj.hours=0
		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss


		else:
			obj.break_hours=0

		
		df_hours.append(obj.hours)
		df_username.append(user.username)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)

	



	df = read_frame(qs)	
	df['hours']=df_hours
	df['username']=df_username
	df["break_hours"]=df_break_hours


	sns.barplot(data=df,x='username',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_employee/1.png')
	plt.close()
	return qs


def total_number_employees():
	qs=User.objects.all()
	return (len(qs) -1)
	# -1 to account for admin 



def employees_present_today():
	today=datetime.date.today()
	qs=Present.objects.filter(date=today).filter(present=True)
	return len(qs)




#used	
def this_week_emp_count_vs_date():
	print("this week vs dte")
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_this_week).filter(date__lte=today)
	str_dates=[]
	emp_count=[]
	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_this_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["Number of employees"]=emp_cnt_all
	
	
	sns.lineplot(data=df,x='date',y='Number of employees')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/this_week/1.png')
	plt.close()






#used
def last_week_emp_count_vs_date():
	today=datetime.date.today()
	some_day_last_week=today-datetime.timedelta(days=7)
	monday_of_last_week=some_day_last_week-  datetime.timedelta(days=(some_day_last_week.isocalendar()[2] - 1))
	monday_of_this_week = monday_of_last_week + datetime.timedelta(days=7)
	qs=Present.objects.filter(date__gte=monday_of_last_week).filter(date__lt=monday_of_this_week)
	str_dates=[]
	emp_count=[]


	str_dates_all=[]
	emp_cnt_all=[]
	cnt=0
	
	



	for obj in qs:
		date=obj.date
		str_dates.append(str(date))
		qs=Present.objects.filter(date=date).filter(present=True)
		emp_count.append(len(qs))


	while(cnt<5):

		date=str(monday_of_last_week+datetime.timedelta(days=cnt))
		cnt+=1
		str_dates_all.append(date)
		if(str_dates.count(date))>0:
			idx=str_dates.index(date)

			emp_cnt_all.append(emp_count[idx])
			
		else:
			emp_cnt_all.append(0)

	
	
	



	df=pd.DataFrame()
	df["date"]=str_dates_all
	df["emp_count"]=emp_cnt_all
	

	
	
	sns.lineplot(data=df,x='date',y='emp_count')
	plt.savefig('./recognition/static/recognition/img/attendance_graphs/last_week/1.png')
	plt.close()


		





# Create your views here.
def home(request):

	return render(request, 'recognition/home.html')

import calendar
def emp_summery(present_qs):
	register_matplotlib_converters()
	df_hours=[]
	df_break_hours=[]

	for obj in qs:
		date=obj.date
		#times_in=time_qs.filter(date=date).filter(out=False).order_by('time')
		#times_out=time_qs.filter(date=date).filter(out=True).order_by('time')
		#times_all=time_qs.filter(date=date).order_by('time')
		#obj.time_in=None
		#obj.time_out=None
		#obj.hours=0
		#obj.break_hours=0
		if (len(times_in)>0):			
			obj.time_in=times_in.first().time
			
		if (len(times_out)>0):
			obj.time_out=times_out.last().time

		if(obj.time_in is not None and obj.time_out is not None):
			ti=obj.time_in
			to=obj.time_out
			hours=((to-ti).total_seconds())/3600
			obj.hours=hours
		

		else:
			obj.hours=0

		(check,break_hourss)= check_validity_times(times_all)
		if check:
			obj.break_hours=break_hourss
		else:
			obj.break_hours=0

		df_hours.append(obj.hours)
		df_break_hours.append(obj.break_hours)
		obj.hours=convert_hours_to_hours_mins(obj.hours)
		obj.break_hours=convert_hours_to_hours_mins(obj.break_hours)
			
	df = read_frame(qs)	
	df["hours"]=df_hours
	df["break_hours"]=df_break_hours

	print(df)
	
	sns.barplot(data=df,x='date',y='hours')
	plt.xticks(rotation='vertical')
	rcParams.update({'figure.autolayout': True})
	plt.tight_layout()
	if(admin):
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/hours_vs_date/1.png')
		plt.close()
	else:
		plt.savefig('./recognition/static/recognition/img/attendance_graphs/employee_login/1.png')
		plt.close()
	return qs
	

@login_required
def dashboard(request):
	user=request.user.username
	if user=="admin": 
		print("admin")
		
		total_num_of_emp=total_number_employees()
		emp_present_today=employees_present_today()
		this_week_emp_count_vs_date()
		last_week_emp_count_vs_date()
		return render(request,"recognition/admin_dashboard.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})

		#return render(request, 'recognition/admin_dashboard.html')
	else:
		print("not admin")

		user=request.user
		today=datetime.date.today()

		y=datetime.date.today().strftime('%Y')
		m=datetime.date.today().strftime('%m')
	
		num_day=calendar.monthrange(int(y),int(m))
		
		start_dt = date(int(y),int(m),1)
		end_dt = datetime.date.today()

		h=Holiday.objects.filter(date__gte=start_dt).filter(date__lt=end_dt)
		hd=[]
		for i in h:
			hd.append(str(i.date))

		start=Present.objects.filter(user=user)
		start=start.filter(date__gte=start_dt).filter(date__lte=end_dt).order_by('-date')
		p=Present.objects.filter(user=user).filter(date__gte=start_dt).filter(present=True).count()
		num_days=int(datetime.date.today().strftime('%d'))
		a=int(num_days)-int(p)
	#	ti=Time.objects.filter(user=user).filter(date=today)
		#qs=hours_vs_employee_given_date(start,ti)

		wo=0
		ho=0
		for dt in daterange(start_dt, end_dt):
			y=dt.year
			m=dt.month
			d=dt.day
			day=date(y,m,d)
			if calendar.day_name[day.weekday()]=="Sunday":
				wo+=1
			if str(day) in hd:
				ho+=1
		a=a-wo
		sd=p+wo
		status=""
		#today=datetime.date.today()
		if start.filter(date=today).exists():
			status="1"
		#times_in=time_qs.filter(user=user).filter(out=False)
		ti=Time.objects.filter(user=user).filter(date=today).filter(out=False).order_by('time')
		to=Time.objects.filter(user=user).filter(date=today).filter(out=True).order_by('-time')
		if ti:
			ti=ti.first().time.time()
		if to:
			to=to.first().time.time()
			
		return render(request,'recognition/employee_dashboard.html',{'user':user,'ds':start_dt,'df':end_dt,'p':p,'a':a,'wo':wo,'sd':sd,'td':num_days,'ho':ho,'today':today,'status':status,'ti':ti,'to':to})


def daterange(date1, date2):
    for n in range(int ((date2 - date1).days)+1):
        yield date1 + timedelta(n)

@login_required
def add_photos(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			create_dataset(username)
			messages.success(request, f'Dataset Created')
			return redirect('add-photos')
		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			return redirect('dashboard')


	else:
		

			form=usernameForm()
			return render(request,'recognition/add_photos.html', {'form' : form})

"""
@login_required
def delete_user(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	if request.method=='POST':
		form=usernameForm(request.POST)
		data = request.POST.copy()
		username=data.get('username')
		if username_present(username):
			#create_dataset(username)
			User.objects.filter(username=username).delete()
			User_details.objects.filter(username=username).delete()
			Present.objects.filter(username=username).delete()
			Time.objects.filter(username=username).delete()

			messages.success(request, f'User has been successfully deleted...')
			form=usernameForm()
			#return render(request,'recognition/delete_user.html', {'form' : form})
			return redirect('dashboard')

		else:
			messages.warning(request, f'No such username found. Please register employee first.')
			form=usernameForm()
			return redirect(request,'recognition/delete_user.html', {'form' : form})



	else:

			form=usernameForm()
			return render(request,'recognition/delete_user.html', {'form' : form})


"""
def mark_your_attendance(request):
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   
	svc_save_path="face_recognition_data/svc.sav"
	
	with open(svc_save_path, 'rb') as f:
		svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')

	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False
	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	person_name=""
	i=0
	flag=False
	frame=[]
	

	while(True):
		
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		#cv2.rectangle(frame,(100,100),(400,400),(0,255,0))

		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)	
		faces = detector(gray_frame,0)
		
		for face in faces:
			print("INFO : inside for loop")

			i+=1
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)


			#if x>100 and y>100 and x+w<500:
			(pred,prob)=predict(face_aligned,svc)
			
			if(pred!=[-1]):
					
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1
				if count[pred] == 4 and (time.time()-start[pred]) > 1.2:
					count[pred] = 0
				else:
					#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
					flag=True
					snap=frame
				
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
						

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance - In - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
		if flag:
			break
	
	vs.stop()
	#filename = 'C:/Users/patel/Desktop/Face-Recognition-system  V1.3/snap/' + person_name + \
	#	str(datetime.datetime.timestamp(datetime.datetime.now()))+'.jpg'
	#cv2.imwrite(filename, img=frame)
	cv2.destroyAllWindows()

	if person_name!='unknown':
		update_attendance_in_db_in(present)
		msg = "Welcome !!!"+person_name
		return render(request,"recognition/confirmation.html",{'user':msg})

	return redirect('home')



def mark_your_attendance_out(request):

	detector = dlib.get_frontal_face_detector()
	
	predictor = dlib.shape_predictor('face_recognition_data/shape_predictor_68_face_landmarks.dat')   
	svc_save_path="face_recognition_data/svc.sav"

	with open(svc_save_path, 'rb') as f:
			svc = pickle.load(f)
	fa = FaceAligner(predictor , desiredFaceWidth = 96)
	encoder=LabelEncoder()
	encoder.classes_ = np.load('face_recognition_data/classes.npy')


	faces_encodings = np.zeros((1,128))
	no_of_faces = len(svc.predict_proba(faces_encodings)[0])
	count = dict()
	present = dict()
	log_time = dict()
	start = dict()
	for i in range(no_of_faces):
		count[encoder.inverse_transform([i])[0]] = 0
		present[encoder.inverse_transform([i])[0]] = False

	vs = VideoStream(src=0).start()
	
	sampleNum = 0
	person_name=""
	while(True):
		
		frame = vs.read()
		frame = imutils.resize(frame ,width = 800)
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = detector(gray_frame,0)
		flag=False

		for face in faces:
			print("INFO : inside for loop")
			(x,y,w,h) = face_utils.rect_to_bb(face)

			face_aligned = fa.align(frame,gray_frame,face)
			cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
			(pred,prob)=predict(face_aligned,svc)
	
			if(pred!=[-1]):
				
				person_name=encoder.inverse_transform(np.ravel([pred]))[0]
				pred=person_name
				if count[pred] == 0:
					start[pred] = time.time()
					count[pred] = count.get(pred,0) + 1

				if count[pred] == 4 and (time.time()-start[pred]) > 1.5:
					 count[pred] = 0
				else:
				#if count[pred] == 4 and (time.time()-start) <= 1.5:
					present[pred] = True
					log_time[pred] = datetime.datetime.now()
					count[pred] = count.get(pred,0) + 1
					print(pred, present[pred], count[pred])
					flag=True
				cv2.putText(frame, str(person_name)+ str(prob), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			else:
				person_name="unknown"
				cv2.putText(frame, str(person_name), (x+6,y+h-6), cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)

			
			
			#cv2.putText()
			# Before continuing to the next loop, I want to give it a little pause
			# waitKey of 100 millisecond
			#cv2.waitKey(50)

		#Showing the image in another window
		#Creates a window with window name "Face" and with the image img
		cv2.imshow("Mark Attendance- Out - Press q to exit",frame)
		#Before closing it we need to give a wait command, otherwise the open cv wont work
		# @params with the millisecond of delay 1
		#cv2.waitKey(1)
		#To get out of the loop
		key=cv2.waitKey(50) & 0xFF
		if(key==ord("q")):
			break
		if flag:
			break
	
	#Stoping the videostream
	vs.stop()

	# destroying all the windows
	cv2.destroyAllWindows()
	if person_name!="unknown":
		update_attendance_in_db_out(present)
		msg = "Bye !!!"+person_name
		return render(request,"recognition/confirmation.html",{'user':msg})
	return render(request,"recognition/home.html")




@login_required
def train(request):
	if request.user.username!='admin':
		return redirect('not-authorised')

	#training_dir='C:/Users/patel/Desktop/Attendance-System-Face-Recognition-master/face_recognition_data/training_dataset'
	training_dir='face_recognition_data/training_dataset'
	
	
	
	count=0
	for person_name in os.listdir(training_dir):
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			count+=1

	X=[]
	y=[]
	i=0


	for person_name in os.listdir(training_dir):
		print(str(person_name))
		curr_directory=os.path.join(training_dir,person_name)
		if not os.path.isdir(curr_directory):
			continue
		for imagefile in image_files_in_folder(curr_directory):
			print(str(imagefile))
			image=cv2.imread(imagefile)
			try:
				X.append((face_recognition.face_encodings(image)[0]).tolist())
				y.append(person_name)
				i+=1
			except:
				print("removed")
				os.remove(imagefile)

			


	targets=np.array(y)
	encoder = LabelEncoder()
	encoder.fit(y)
	y=encoder.transform(y)
	X1=np.array(X)
	print("shape: "+ str(X1.shape))
	np.save('face_recognition_data/classes.npy', encoder.classes_)
	svc = SVC(kernel='linear',probability=True)
	svc.fit(X1,y)
	svc_save_path="face_recognition_data/svc.sav"
	with open(svc_save_path, 'wb') as f:
		pickle.dump(svc,f)

	
	vizualize_Data(X1,targets)
	
	messages.success(request, f'Training Complete.')

	return render(request,"recognition/train.html")


@login_required
def not_authorised(request):
	return render(request,'recognition/not_authorised.html')



@login_required
def view_attendance_home(request):
	total_num_of_emp=total_number_employees()
	emp_present_today=employees_present_today()
	this_week_emp_count_vs_date()
	last_week_emp_count_vs_date()
	return render(request,"recognition/view_attendance_home.html", {'total_num_of_emp' : total_num_of_emp, 'emp_present_today': emp_present_today})
	#return render(request,"recognition/view_attendance_home.html")

@login_required
def view_attendance_date(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None


	if request.method=='POST':
		form=DateForm(request.POST)
		if form.is_valid():
			date=form.cleaned_data.get('date')
			date1=form.cleaned_data.get('date')

			print("date:"+ str(date))
			time_qs=Time.objects.filter(date=date)
			present_qs=Present.objects.filter(date=date)
			if(len(time_qs)>0 or len(present_qs)>0):
				qs=hours_vs_employee_given_date(present_qs,time_qs)


				y=date1.strftime('%Y')
				m=date1.strftime('%m')
				d=date1.strftime('%d')

				#start_dt = date(int(y),int(m),int(d))
				u=[]
				for i in qs:
					u.append(i.user.username)
				print(qs)
		
				l=[]
				for i in qs:
					if i.user.username in u:
						print("present : ",i.user.username)
						a=ls(str(i.date),i.user.username,"P",i.time_in,i.time_out,i.hours,i.break_hours)
						l.append(a)

				total_emp=User.objects.order_by('-username')
				for i in total_emp:		
					if i.username not in u:
						a=ls(date,i.username,"A","-","-","-","-")
						l.append(a)

				l.sort(key = lambda x : str(x.d))

				#for i in l:
				#	print("list :",i.e)
				
				return render(request,'recognition/view_attendance_date.html', {'form' : form,'qs' : l })
			else:
				messages.warning(request, f'No records for selected date.')
				return redirect('view-attendance-date')

	else:
		

			form=DateForm()
			return render(request,'recognition/view_attendance_date.html', {'form' : form, 'qs' : qs})


@login_required
def view_attendance_employee(request):
	if request.user.username!='admin':
		return redirect('not-authorised')
	time_qs=None
	present_qs=None
	qs=None

	if request.method=='POST':
		form=UsernameAndDateForm(request.POST)
		if form.is_valid():
			username=form.cleaned_data.get('username')
			if username_present(username):
				
				u=User.objects.get(username=username)
				
				time_qs=Time.objects.filter(user=u)
				present_qs=Present.objects.filter(user=u)
				date_from=form.cleaned_data.get('date_from')
				date_to=form.cleaned_data.get('date_to')
				
				if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-attendance-employee')
				else:

					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					
					snap_in=[]
					snap_out=[]
					
					"""for i in time_qs:
						if i.out==False:
							snap_in.append(i.image)
						else:
							snap_out.append(i.image)"""
					#print(snap_in)

					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=True)

						l=[]
						days=[]
						x=0
						for i in qs:
							if i.present:
								if x<len(snap_in):
									a=ls(str(i.date),i.user.username,"P",i.time_in,i.time_out,i.hours,i.break_hours)
								else:
									a=ls(str(i.date),i.user.username,"P",i.time_in,i.time_out,i.hours,i.break_hours)
							else:
								a=ls(str(i.date),i.user.username,"A",i.time_in,i.time_out,i.hours,i.break_hours)
							l.append(a)
							days.append(str(a.d))
							#print("\n",a.d)
						num_days=int(datetime.date.today().strftime('%d'))
						h=Holiday.objects.filter(date__gte=date_from).filter(date__lt=date_to)
						hd=[]
						for i in h:
							hd.append(str(i.date))	

						y=date_from.strftime('%Y')
						m=date_from.strftime('%m')
						d=date_from.strftime('%d')

						yf=date_to.strftime('%Y')
						mf=date_to.strftime('%m')
						df=date_to.strftime('%d')

						start_dt = date(int(y),int(m),int(d))
						end_dt = date(int(yf),int(mf),int(df))
						i=0
						ho=0
						for dt in daterange(start_dt, end_dt):
							#a=l[i]
							y=dt.year
							m=dt.month
							d=dt.day
							day=str(date(y,m,d))
							d1=date(y,m,d)
							#print("\n",day,a.d)
							#print("\n",type(day),type(days))
							if day not in days:
								#print(day)
								if calendar.day_name[d1.weekday()]=="Sunday":
									a=ls(str(day),u,"WO","-","-","-","-")
								if day in hd:
									a=ls(str(day),u,"HO","-","-","-","-")	
									ho+=1
									#print("holiday :" ,day)
								else :
									a=ls(str(day),u,"A","-","-","-","-")
								
								l.append(a)
							i=i+1
						
						l.sort(key = lambda x : str(x.d))

						user=request.user
						today=datetime.date.today()

						#y=datetime.date.today().strftime('%Y')
						#m=datetime.date.today().strftime('%m')
						y=date_from.strftime('%Y')
						m=date_from.strftime('%m')
						d=date_from.strftime('%d')

						#num_day=calendar.monthrange(int(y),int(m))
						date_start=datetime.date.today().strftime('%Y-%m-01')
						date_fin=datetime.date.today().strftime('%Y-%m-%d')

						yf=date_to.strftime('%Y')
						mf=date_to.strftime('%m')
						df=date_to.strftime('%d')

						start_dt = date(int(y),int(m),int(d))
						end_dt = date(int(yf),int(mf),int(df))
						num_days=end_dt-start_dt
						num_days=num_days.days+1
						
						start=Present.objects.filter(user=user)
						start=start.filter(date__gte=start_dt).filter(date__lte=end_dt).order_by('-date')
						#p=start.filter(present=True).count()
						p=present_qs.filter(date__gte=start_dt).filter(date__lte=end_dt).order_by('-date').count()
						a=int(num_days)-int(p)
						

						
						wo=0
						for dt in daterange(start_dt, end_dt):
							y=dt.year
							m=dt.month
							d=dt.day
							day=date(y,m,d)
							if calendar.day_name[day.weekday()]=="Sunday":
								print(day)
								wo+=1
						a=a-wo	
						sd=p+wo

						return render(request,'recognition/view_attendance_employee.html', {'form' : form, 'qs' :l,'p':p,'a':a,'wo':wo,'sd':sd,'ho':ho})
					else:
						#print("inside qs is None")
						messages.warning(request, f'No records for selected duration.')
						notfound=1
						return render(request,'recognition/view_attendance_employee.html', {'form' : form,'nf':notfound})

				
			else:
				print("invalid username")
				messages.warning(request, f'No such username found.')
				return redirect('view-attendance-employee')


	else:

			form=UsernameAndDateForm()
			#return redirect('view-attendance-employee')

			return render(request,'recognition/view_attendance_employee.html', {'form' : form})

class ls:
	def __init__(self,d,e,p,ti,to,h,bh):
		self.d=d
		self.e=e
		self.p=p
		self.ti=ti
		self.to=to
		self.h=h
		self.bh=bh
		


@login_required
def view_my_attendance_employee_login(request):
	if request.user.username=='admin':
		return redirect('not-authorised')
	qs=None
	time_qs=None
	present_qs=None
	u=request.user

	if request.method=='POST':
		form=DateForm_2(request.POST)
		if form.is_valid():
			time_qs=Time.objects.filter(user=u)
			present_qs=Present.objects.filter(user=u)
			date_from=form.cleaned_data.get('date_from')
			date_to=form.cleaned_data.get('date_to')
			if date_to < date_from:
					messages.warning(request, f'Invalid date selection.')
					return redirect('view-my-attendance-employee-login')
			else:
					
					time_qs=time_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
					present_qs=present_qs.filter(date__gte=date_from).filter(date__lte=date_to).order_by('-date')
				
					if (len(time_qs)>0 or len(present_qs)>0):
						qs=hours_vs_date_given_employee(present_qs,time_qs,admin=False)
						l=[]
						days=[]
						for i in qs:
							if i.present:
								a=ls(str(i.date),i.user.username,"P",i.time_in,i.time_out,i.hours,i.break_hours)
							else:
								a=ls(str(i.date),i.user.username,"A",i.time_in,i.time_out,i.hours,i.break_hours)
							l.append(a)
							days.append(str(a.d))
							#print("\n",a.d)
						num_days=int(datetime.date.today().strftime('%d'))
						h=Holiday.objects.filter(date__gte=date_from).filter(date__lte=date_to)
						hd=[]
						for i in h:
							hd.append(str(i.date))

						print("holiday list :",hd)
						#y=datetime.date.today().strftime('%Y')
						y=date_from.strftime('%Y')
						#m=datetime.date.today().strftime('%m')
						m=date_from.strftime('%m')
						d=date_from.strftime('%d')

						yf=date_to.strftime('%Y')
						mf=date_to.strftime('%m')
						df=date_to.strftime('%d')

						start_dt = date(int(y),int(m),int(d))
						end_dt = date(int(yf),int(mf),int(df))
						i=0
						ho=0
						for dt in daterange(start_dt, end_dt):
							#a=l[i]
							y=dt.year
							m=dt.month
							d=dt.day
							day=str(date(y,m,d))
							d1=date(y,m,d)
							#print("\n",day,a.d)
							#print("\n",type(day),type(days))
							if day not in days:
								if calendar.day_name[d1.weekday()]=="Sunday":
									a=ls(str(day),u,"WO","-","-","-","-")
								if day in hd:
									a=ls(str(day),u,"HO","-","-","-","-")
								else :
									a=ls(str(day),u,"A","-","-","-","-")
								l.append(a)
							i=i+1
						
						#for i in l:
							#print(i.d,i.e,i.p)

						wo=0
						ho=0
						for dt in daterange(start_dt,end_dt):
							y=dt.year
							m=dt.month
							d=dt.day
							day=date(y,m,d)
							if calendar.day_name[day.weekday()]=="Sunday":
								wo+=1
							if str(day) in hd:
								ho+=1 
						
							
						print(type(l))
						l.sort(key = lambda x : str(x.d))

						p=present_qs.filter(date__gte=start_dt).filter(date__lte=end_dt).count()
						n=end_dt-start_dt
						n=n.days+1
						a=int(n)-int(p)-wo-ho
						sd=int(p)+ho+wo


						return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs,'l':l,'p':p,'a':a,'wo':wo,'ho':ho,'sd':sd})
					else:
						
						messages.warning(request, f'No records for selected duration.')
						return redirect('view-my-attendance-employee-login')
	else:
		

			form=DateForm_2()
			return render(request,'recognition/view_my_attendance_employee_login.html', {'form' : form, 'qs' :qs})

@login_required
def view_my_profile(request):
	
	users=request.user
	q=0
	p=0
	u=User_details.objects.filter(user=users).exists()
	if u:
		qs=User_details.objects.get(user=users)
		p=1
	else:
		print("not present")
		#print(users)
		qs=users
		q=1
	print(qs)
		
	return render(request,'recognition/user_profile.html',{'qs':qs,'q':q,'p':p})

from users.models import User_details
class u:
	def __init__(self,user,email,contact,department):
		self.user=user
		self.email=email
		self.contact=contact
		self.department=department

@login_required
def view_employee(request):
	qs=User_details.objects.order_by('user')
	qs1=User.objects.order_by('username')
	
	l1=[]
	l=[]
	x=[]
	y={""}
	
	for i in qs:
		d=u(i.user,i.email,i.contact,i.department)
		l.append(d)
		a=d.user
		print("in 1st:",a)
		x.append(a)

	for i in x:
		y.add(i.username)
		#print("\nin x:",i)
	
	for i in qs1:
		if i.username  in y:
			print('')
			
		else:
			print("not in list",i.username,y)
			d=u(i.username,'','','')

			l.append(d)

	return render(request,'recognition/view_employees.html',{'qs':qs,'l':l})
	#return HttpResponse("jay")

def editprofile(request):

	print("profile edited 1")

	if request.method=="POST":
		user=request.user
		email=request.POST.get('email')
		contact=request.POST.get('contact')
		dept=request.POST.get('department')

		print("\nstart")
		print("\nuser : ",user,"email : ",email,"\ncontact :",contact,"\ndepartment :",dept)

	if User_details.objects.filter(user=user).exists():
		User_details.objects.filter(user=user).update(email=email,contact=contact,department=dept)
		qs=User_details.objects.get(user=user)

		print("exist")
	else:
		User_details.objects.create(user=user,email=email,contact=contact,department=dept)	
		qs=User_details.objects.get(user=user)

		print("not exist")

			
		print("profile edited")

	return render(request,'recognition/user_profile.html',{'p':1,'q':0,'qs':qs})

@login_required
def holiday(request):
	today=datetime.date.today()
	qs=Holiday.objects.filter(date__gte=today).order_by('date')
	
	if request.method=="POST":
		if(request.user.username=='admin'):
			form=Holiday_form(request.POST)
			if form.is_valid():

				name=form.cleaned_data.get('name')
				date=form.cleaned_data.get('date')
				Holiday.objects.create(name=name,date=date)
				
				return render(request, 'recognition\holiday.html',{'form':form,'h':qs})
		else:
			return render('not_authorised')
	else:	
		form=Holiday_form()
		return render(request, 'recognition\holiday.html',{'form':form,'h':qs})

@login_required
def employee_holiday(request):

	today=datetime.date.today()
	qs=Holiday.objects.filter(date__gte=today).order_by('date')
		
	if(request.user.username!='admin'):
		return render(request, 'recognition\employee_holiday.html',{'h':qs})
	else:	
		return render('not_authorised')


	