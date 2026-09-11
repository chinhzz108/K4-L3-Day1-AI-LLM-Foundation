# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**

**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
Khi test với 4 mức nhiệt độ, mình thấy rõ là temperature càng thấp (0.0 và 0.5) thì model trả lời rất chắc chắn, thường chọn sự thật phổ biến nhất như xuất khẩu cà phê và chạy lại thì nội dung hầu như y hệt nhau. Lên mức 1.0 thì câu trả lời bắt đầu phong phú hơn, chuyển sang kể về địa đạo Củ Chi hay Sơn Đoòng với nhiều cảm xúc hơn. Đến mức 1.5 thì câu cú hơi bay bổng quá đà, dùng nhiều từ ngữ hoa mỹ và cảm giác thông tin bắt đầu kém chính xác hơn.

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
Với chatbot chăm sóc khách hàng thì mình sẽ để tầm 0.0 đến khoảng 0.2 thôi. Lý do là bot CSKH cần nhất sự chính xác và đồng nhất, hỏi giá tiền hay chính sách đổi trả thì phải trả lời đúng quy định của công ty chứ không được sáng tạo hay bịa thông tin gây hiểu lầm cho khách.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
Tính ra mỗi ngày hệ thống có 30.000 lượt gọi, tổng cộng tầm 10.5 triệu token output. Tính theo bảng giá thì GPT-4o tốn khoảng $105/ngày còn GPT-4o-mini chỉ mất tầm $6.3/ngày, tức là bản to đắt hơn bản mini tầm 16.7 lần (chênh nhau gần 3.000 đô một tháng). 
Theo mình bản GPT-4o đắt nhưng sẽ đáng tiền khi cần xử lý bài toán khó như đọc hiểu hợp đồng pháp lý, phân tích tài chính sâu hoặc code các thuật toán phức tạp. Còn những việc đơn giản như phân loại ý định người dùng, tóm tắt tin nhắn ngắn hay trả lời mấy câu hỏi thường gặp thì dùng mini là quá đủ và tiết kiệm.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
Hai kết quả ra khác hẳn nhau luôn: khi đóng vai cô giáo tiểu học thì model nói chuyện rất gần gũi, lấy ví dụ dễ thương như cuốn sổ chung của lớp để các bạn chia đồ chơi; còn khi đổi sang chuyên gia tài chính thì câu trả lời đầy thuật ngữ chuyên ngành như DLT, sổ cái phân tán với mạng ngang hàng. Mình thấy system prompt tác động cực kỳ mạnh đến cách hành xử của model. Nó giống như việc nhập vai, quyết định luôn giọng điệu, mức độ phức tạp của từ vựng và đối tượng tiếp nhận mà không cần phải đổi câu hỏi chính.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
Mình test thử một đoạn văn tiếng Việt 125 từ về AI, đếm bằng tiktoken ra 138 token còn công thức ước lượng lấy số từ chia 0.75 thì ra khoảng 166.7 token, lệch nhau cỡ 20%. Tiếng Việt mình thường tốn nhiều token hơn tiếng Anh vì các bộ tokenizer hiện tại (dùng Byte-Pair Encoding) chủ yếu được train trên dữ liệu tiếng Anh, từ vựng tiếng Anh thường nằm trọn trong 1 token. Trong khi đó chữ tiếng Việt có nhiều dấu câu và nguyên âm ghép (như ă, â, ư, ơ...) là các ký tự Unicode nhiều byte, nên tokenizer hay phải chặt nhỏ một từ ra thành nhiều mảnh sub-word hoặc byte lẻ để mã hóa.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
Streaming ăn điểm nhất là ở các giao diện chat trực tiếp với người dùng, vì nó giúp chữ hiện ra ngay lập tức (giảm thời gian chờ token đầu tiên) làm người dùng thấy hệ thống phản hồi nhanh và đỡ sốt ruột khi câu trả lời dài. Ngược lại, non-streaming sẽ hợp hơn với các tác vụ chạy ngầm phía server, ví dụ như xử lý dữ liệu hàng loạt theo lô (batch), các API giao tiếp giữa backend với backend, hoặc khi cần mô hình trả về định dạng JSON chuẩn để code validate và parse toàn bộ dữ liệu một thể.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
Khi API nghẽn mạng thì exponential backoff thông minh hơn nhiều vì nó cho thời gian chờ tăng dần theo cấp số nhân (chờ 1s rồi 2s, 4s, 8s...), tạo khoảng nghỉ ngày càng rộng để server kịp giải tỏa tải. Nếu tất cả client đều cài delay cố định ví dụ đúng 1 giây thì sau mỗi giây cả nghìn máy sẽ cùng lúc dội request ngược lại vào server. Hiện tượng này gọi là bão retry (thundering herd), sẽ làm server sập liên tục và không bao giờ ngóc đầu dậy nổi.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
Mình chọn persona làm một bạn trợ giảng AI dạy kèm lập trình Python cho người mới.
System prompt mình viết: "Bạn là một trợ giảng AI vui vẻ, kiên nhẫn chuyên giải đáp bài tập Python cho sinh viên. Hãy giải thích ngắn gọn, dễ hiểu, kèm ví dụ code ngắn và luôn trả lời bằng tiếng Việt trong khoảng 3-4 câu."
Trong prompt này mình có 2 lưu ý quan trọng:
- Thứ nhất là chốt cứng "luôn trả lời bằng tiếng Việt": vì khi hỏi về thuật ngữ lập trình tiếng Anh, model rất dễ bị lái sang trả lời hoàn toàn bằng tiếng Anh nếu không nhắc trước.
- Thứ hai là "ngắn gọn trong khoảng 3-4 câu": để câu trả lời hiện lên terminal gọn gàng, người đọc không bị ngợp chữ và cũng đỡ tốn token output của API.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
Con bot hiện tại của mình điểm yếu lớn nhất là não cá vàng, chỉ nhớ được 3 lượt chat gần nhất rồi tự động cắt bỏ phần trước, tắt terminal đi là quên sạch mọi thứ.
Để cải thiện, mình đề xuất làm thêm cơ chế tóm tắt lịch sử (summary) và lưu vào file SQLite hoặc JSON local. Cách làm là: khi lịch sử chạm ngưỡng 6 tin nhắn, mình cho một model nhỏ như GPT-4o-mini tóm lược các trao đổi cũ thành 1-2 câu tóm tắt rồi gắn đè vào sau system prompt; đồng thời ghi lịch sử xuống file để lần sau người dùng mở máy lên bot vẫn đọc lại được ngữ cảnh cũ mà không bị tràn context.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã copy bài làm vào folder `solution/`, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
