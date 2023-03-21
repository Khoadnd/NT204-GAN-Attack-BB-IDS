col_names_org = ["duration","protocol_type","service","flag","src_bytes",
             "dst_bytes","land","wrong_fragment","urgent","hot","num_failed_logins",
             "logged_in","num_compromised","root_shell","su_attempted","num_root",
             "num_file_creations","num_shells","num_access_files","num_outbound_cmds",
             "is_host_login","is_guest_login","count","srv_count","serror_rate",
             "srv_serror_rate","rerror_rate","srv_rerror_rate","same_srv_rate",
             "diff_srv_rate","srv_diff_host_rate","dst_host_count","dst_host_srv_count",
             "dst_host_same_srv_rate","dst_host_diff_srv_rate","dst_host_same_src_port_rate",
             "dst_host_srv_diff_host_rate","dst_host_serror_rate","dst_host_srv_serror_rate",
             "dst_host_rerror_rate","dst_host_srv_rerror_rate","label","level"]

attack_category_dict = {
    1:['apache2','back','land','neptune','mailbomb','pod','processtable','smurf','teardrop','udpstorm','worm'], #dos
    2:['ipsweep','mscan','nmap','portsweep','saint','satan'], #u2r
    3:['buffer_overflow','loadmodule','perl','ps','rootkit','sqlattack','xterm'], #r2l
    4:['ftp_write','guess_passwd','httptunnel','imap','multihop','named','phf','sendmail','snmpgetattack','spy','snmpguess','warezclient','warezmaster','xlock','xsnoop'] #probe
}

features_binary = ['land', 'logged_in', 'root_shell', 'is_host_login', 'is_guest_login','urgent','wrong_fragment', 'su_attempted']
features_category = ["protocol_type","service","flag"]
