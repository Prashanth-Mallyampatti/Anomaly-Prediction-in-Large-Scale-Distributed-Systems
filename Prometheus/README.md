# Prometheus

Installation and Setup of Prometheus

<br />Execute the following commands to set up prometheus:

    sudo useradd --no-create-home --shell /bin/false prometheus
    
    sudo mkdir /etc/prometheus
    
    sudo mkdir /var/lib/prometheus
    
    sudo chown prometheus:prometheus /etc/prometheus
    
    sudo chown prometheus:prometheus /var/lib/prometheus
    
    curl -LO https://github.com/prometheus/prometheus/releases/download/v2.9.1/prometheus-2.9.1.linux-amd64.tar.gz
    
    tar xvf prometheus-2.9.1.linux-amd64.tar.gz
    
<br />After every command execution is successful, you'll get output something like this:
    
    e12917b25b32980daee0e9cf879d9ec197e2893924bd1574604eb0f550034d46  prometheus-2.9.1.linux-amd64.tar.gz

<br />Unpack the downloaded archive and execute the rest:

    tar xvf prometheus-2.9.1.linux-amd64.tar.gz
    
    sudo cp prometheus-2.9.1.linux-amd64/prometheus /usr/local/bin/
    
    sudo cp prometheus-2.9.1.linux-amd64/promtool /usr/local/bin/
    
    sudo chown prometheus:prometheus /usr/local/bin/prometheus
    
    sudo chown prometheus:prometheus /usr/local/bin/promtool
    
    sudo cp -r prometheus-2.9.1.linux-amd64/consoles /etc/prometheus/
    
    sudo cp -r prometheus-2.9.1.linux-amd64/console_libraries/ /etc/prometheus/
    
    sudo chown -R prometheus:prometheus /etc/prometheus/consoles
    
    sudo chown -R prometheus:prometheus /etc/prometheus/console_libraries
    
    rm -rf prometheus-2.9.1.linux-amd64.tar.gz prometheus-2.9.1.linux-amd64
    
<br /> Now, "prometheus.yml" as used in the command below can be found here(its in this directory): [prometheus.yml](https://github.ncsu.edu/pmallya/Anomaly-Prediction-in-Large-Scale-Distributed-Systems/blob/master/Prometheus/prometheus.yml)

    cp prometheus.yml /etc/prometheus/
    
    sudo chown prometheus:prometheus /etc/prometheus/prometheus.yml
    
    
    
<br />Running Prometheus in background:

    sudo -u prometheus /usr/local/bin/prometheus \
        --config.file /etc/prometheus/prometheus.yml \
        --storage.tsdb.path /var/lib/prometheus/ \
        --web.console.templates=/etc/prometheus/consoles \
        --web.console.libraries=/etc/prometheus/console_libraries
        --storage.tsdb.retention.time=200d 2>&1 &
        
<br /> The output contains information about Prometheus' loading progress, configuration file, and related services. It also confirms that Prometheus is listening on port `9090`.
<br />Now, halt Prometheus by pressing `CTRL+C`, and then open a new `systemd` service file.

    sudo nano /etc/systemd/system/prometheus.service
    
<br />The service file tells `systemd` to run Prometheus as the prometheus user, with the configuration file located in the `/etc/prometheus/prometheus.yml` directory and to store its data in the `/var/lib/prometheus` directory.

<br />Copy the following content into the Prometheus service file - `/etc/systemd/system/prometheus.service`:
    
    [Unit]
    Description=Prometheus
    Wants=network-online.target
    After=network-online.target

    [Service]
    User=prometheus
    Group=prometheus
    Type=simple
    ExecStart=/usr/local/bin/prometheus \
        --config.file /etc/prometheus/prometheus.yml \
        --storage.tsdb.path /var/lib/prometheus/ \
        --web.console.templates=/etc/prometheus/consoles \
        --web.console.libraries=/etc/prometheus/console_libraries

    [Install]
    WantedBy=multi-user.target
    
<br />Save the file and close the file. To use the newly created service, reload `systemd` and start `prometheus`.

    sudo systemctl daemon-reload
    
    sudo systemctl start prometheus
    
<br />Prometheus status check: The output tells you Prometheus' status, main process identifier (PID), memory use, and more.
    
    sudo systemctl status prometheus


<br />Enabling Prometheus service to start on boot:

    sudo systemctl enable prometheus
