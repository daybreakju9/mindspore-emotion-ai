<template>
  <div class="chat-container">
    <!-- 消息区 -->
    <div class="messages" ref="messagesRef">
      <div
        v-for="(msg, index) in messages"
        :key="index"
        :class="['message', msg.role]"
      >
        <div class="bubble">
          {{ msg.content }}
        </div>
      </div>
    </div>

    <!-- 输入区 -->
    <div class="input-area">
      <el-input
        v-model="inputText"
        placeholder="输入你的消息..."
        type="textarea"
        :rows="2"
        @keyup.enter.exact.prevent="sendMessage"
      />
      <el-button type="primary" @click="sendMessage" style="margin-left: 10px;">
        发送
      </el-button>
    </div>
  </div>
</template>

<script setup>
import { ref, nextTick } from 'vue'

// 聊天记录
const messages = ref([
  { role: 'ai', content: '你好，我是你的情绪分析助手！' }
])

// 输入框内容
const inputText = ref('')

// 自动滚动到底部
const messagesRef = ref(null)
const scrollToBottom = () => {
  nextTick(() => {
    if (messagesRef.value) {
      messagesRef.value.scrollTop = messagesRef.value.scrollHeight
    }
  })
}

// 发送消息
const sendMessage = () => {
  if (!inputText.value.trim()) return

  // 用户消息加入列表
  messages.value.push({
    role: 'user',
    content: inputText.value
  })

  // 清空输入框
  inputText.value = ''

  scrollToBottom()

  // 模拟 AI 回复（之后这里改成你们真实后端）
  setTimeout(() => {
    messages.value.push({
      role: 'ai',
      content: '我收到你的消息啦！等后端接好了这里会变成真正的 AI 回复。'
    })
    scrollToBottom()
  }, 600)
}
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: 100vh;
  padding: 15px;
  box-sizing: border-box;
}

/* 消息列表区域 */
.messages {
  flex: 1;
  overflow-y: auto;
  padding-right: 10px;
}

/* 气泡布局：左 AI / 右 用户 */
.message {
  display: flex;
  margin-bottom: 12px;
}

.message.user {
  justify-content: flex-end;
}

.message.ai {
  justify-content: flex-start;
}

/* 气泡样式 */
.bubble {
  max-width: 70%;
  padding: 10px 14px;
  border-radius: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
}

/* 用户气泡（绿色风格） */
.user .bubble {
  background-color: #4caf50;
  color: white;
  border-bottom-right-radius: 2px;
}

/* AI 氣泡（浅灰风格） */
.ai .bubble {
  background-color: #e5e5e5;
  color: #333;
  border-bottom-left-radius: 2px;
}

/* 底部输入栏 */
.input-area {
  display: flex;
  align-items: center;
  padding: 10px 0;
}
</style>

