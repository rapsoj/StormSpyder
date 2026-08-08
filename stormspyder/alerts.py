from datetime import datetime

import pandas as pd

try:
    import yagmail
except ImportError:  # pragma: no cover - exercised in minimal environments
    yagmail = None


def get_key_by_value(d, value):
    for key, val in d.items():
        if val == value:
            return key


def format_regions(arr):
    length = len(arr)
    if length == 0:
        return 'unclaimed territories'
    if length == 1:
        return arr[0]
    if length == 2:
        return f'{arr[0]} and {arr[1]}'
    if length == 3:
        return f'{arr[0]}, {arr[1]}, and {arr[2]}'
    if length == 4:
        return f'{arr[0]}, {arr[1]}, {arr[2]}, and one other country'
    return f'{arr[0]}, {arr[1]}, {arr[2]}, and {length - 3} other countries'


def format_impact(num):
    if num > 1000000:
        return str(round(num / 1000000, 1)) + 'M'
    if num > 1000:
        return str(round(num / 1000, 1)) + 'K'
    return int(round(num, 0))


def format_alert(df_all, n):
    n_impact = df_all.sort_values('expected_impact', ascending=False).iloc[n]
    storm_type = n_impact['storm_type']
    regions = format_regions(n_impact['regions'])
    impact = format_impact(n_impact['expected_impact'])
    date = n_impact['date']
    return f"{storm_type.capitalize()} alert: {impact} estimated impacted in {regions} on {date}"


def send_email_alert(df_all, storm_type_options):
    if df_all.empty:
        print('No alert data available; skipping email send.')
        return

    if yagmail is None:
        raise RuntimeError('yagmail is required to send email alerts')

    today_date = datetime.today().strftime('%Y-%m-%d')
    data = df_all.sort_values('expected_impact', ascending=False)
    data = data[data['expected_impact'] > 0]
    data['regions'] = data['regions'].apply(lambda x: ', '.join(x))
    data['reported'] = today_date
    data.to_csv(f'{today_date}-storm-alert.csv', index=False)

    top_impact = df_all.sort_values('expected_impact', ascending=False).iloc[0]
    storm_type = top_impact['storm_type']
    regions = format_regions(top_impact['regions'])
    impact = format_impact(top_impact['expected_impact'])
    date = top_impact['date']

    FROM = {'stormspyder.alerts@gmail.com': 'StormSpyder'}
    TO = ['jessicakristenr@gmail.com', 'elisabeth.stephens@reading.ac.uk']
    APP_PASSWORD = 'blsz ardo uicy qfnx'
    SUBJECT = f"{storm_type.capitalize()} alert: {impact} estimated impacted in {regions} on {date}"
    IMAGE_PATH = 'temp/' + get_key_by_value(storm_type_options, storm_type) + '/' + date + '.png'
    CSV_PATH = f'{today_date}-storm-alert.csv'

    alert_list = '<ol>'
    for n in range(len(df_all)):
        alert = format_alert(df_all, n)
        alert_list += f'<li>{alert}</li>'
    alert_list += '</ol>'

    message_html = f"""
    <html>
      <body>
        <p>Dear recipient,</p>
        <p>The following tropical storm events have been forecast by the ECMWF:</p>
        {alert_list}
        <p>{storm_type.capitalize()} forecast for {date}:</p>
        <img src="cid:image.png">
        <p>To learn more about how this tool works, <a href="https://github.com/rapsoj/StormSpyder/tree/main">click here</a>.</p>
        <p>To unsubscribe from future alerts, send a message to <a href="mailto:stormspyder.alerts@gmail.com?subject=UNSUBSCRIBE">stormspyder.alerts@gmail.com</a> with the subject line "UNSUBSCRIBE".</p>
        <p>Best regards,<br>StormSpyder</p>
      </body>
    </html>
    """

    yag = yagmail.SMTP(FROM, APP_PASSWORD)
    yag.send(
        bcc=TO,
        subject=SUBJECT,
        contents=[message_html, yagmail.inline(IMAGE_PATH)],
        attachments=CSV_PATH,
    )
    print('---Email sent successfully---')

    record = pd.read_csv('record.csv')
    if not record['reported'].isin([today_date]).any():
        record = pd.concat([record, data], ignore_index=True)
        record.to_csv('record.csv', index=False)

    import os
    os.remove(CSV_PATH)
