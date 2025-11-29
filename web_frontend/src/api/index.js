import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:5030/api', // 和后端商量或查看 swagger
  timeout: 5000
})

export default api

