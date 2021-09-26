
#!/bin/bash/python3

# Import requests for API and tkinter for GUI
import requests
from tkinter import *


# Define request function
def request_response(text):
    query = {'input':text}
    response = requests.put('http://3.235.0.255:5000', params=query)
    # response = requests.put('http://67.202.54.165:5000', params=query)
    return response.json()['body']


# Define functionality of send button.
def send(event=None):
    msg = EntryBox.get("1.0",'end-1c').strip()
    EntryBox.delete("0.0",END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n', ('you_fg'))

        res = request_response(msg)
        ChatLog.insert(END, "Cogswell: " + res + '\n\n', ('bot_fg'))

        ChatLog.config(state=DISABLED)
        ChatLog.yview(END)


# Create base tkinter GUI
root = Tk()
root.title("Cogswell")
root.geometry("400x500")
root.resizable(width=FALSE, height=FALSE)
root.configure(bg='#1C6353')
root.bind('<Return>', send)


# Build the chat window
ChatLog = Text(root, bd=2, bg="white", height="8", width="50", font="Arial",)
ChatLog.tag_configure("you_fg", foreground="#000063", font=("Verdana", 12 ))
ChatLog.tag_configure("bot_fg", foreground="#1C6353", font=("Verdana", 12 ))
ChatLog.config(state=DISABLED)


# Bind the scrollbar
scrollbar = Scrollbar(root, command=ChatLog.yview)
ChatLog['yscrollcommand'] = scrollbar.set


photo = PhotoImage(file = r"Cogswell.png")

# Create send button
SendButton = Button(root, font=("Verdana",12,'bold'), width=116, height=90,
                    bd=2, 
                    bg="#4CB3CF",
                    # bg="#3c9d9b",
                    activebackground="#71CDF4", fg='#ffffff',
                    command=send, image=photo)


# Text box for message entry
EntryBox = Text(root, bd=2, bg="white",width="29", height="5", font="Arial")


# Set all pieces in the application space.
scrollbar.place(x=376,y=6, height=386)
ChatLog.place(x=6,y=6, height=386, width=370)
EntryBox.place(x=6, y=401, height=90, width=265)
SendButton.place(x=277, y=401, height=90, width=116)


# Run application loop
root.mainloop()