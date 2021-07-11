import cv2
#cv2:-computer vision

cap= cv2.VideoCapture(0)
#0 means active the camera suppose if we want to open a video can give any video file name instead of zero

faceModel = cv2.CascadeClassifier("ai_face_brain.xml")
smileModel= cv2.CascadeClassifier("haarcascade_smile.xml")
eyeModel=cv2.CascadeClassifier('ai_eye_brain.xml')


while True:
	
	ret, frame = cap.read()
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	faces=faceModel.detectMultiScale(gray,1.3,5)
	

	for (x,y,w,h) in faces:
		
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
		cv2.putText(frame,'Face',(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
		section_gray=gray[y:y+h,x:x+w]
		frame_face=frame[y:y+h,x:x+w]

		smile=smileModel.detectMultiScale(section_gray)
		

		for(sx,sy,sw,sh)in smile:
			cv2.rectangle(section_gray,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
			cv2.putText(frame_face,'smile',(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)


		eyes=eyeModel.detectMultiScale(section_gray)
		for(ex,ey,ew,eh) in eyes:
			cv2.rectangle(section_gray,(ex,ey),(ex+ew,ey+eh),(0,0,255),2)
			cv2.putText(frame_face,'eye',(x+w,y+h),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
			


	cv2.imshow('Face',frame)

	if cv2.waitKey(1)==ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

