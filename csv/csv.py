

import csv
with open('/media/pi/Scandisk-500GB/Daten/Visual-Studio/Phyton/csv/alle.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerows(someiterable)