pipeline {
    agent any

    stages {
        stage('Clone GitHub Repo') {
            steps {
                echo 'Cloning GitHub repo to Jenkins...'
                checkout scm
            }
        }

        stage('Test') {
            steps {
                sh 'pwd'
                sh 'ls -la'
            }
        }
    }
}
