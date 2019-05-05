from slacker import Slacker
def send_to_slack(message):
    slack = Slacker('xoxp-595326285747-611017066343-599631974547-22274301c83aa45f60b590f58eeda98e')
    slack.chat.post_message('@rnarang', message)
