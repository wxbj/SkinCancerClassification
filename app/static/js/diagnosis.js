document.getElementById("userForm").addEventListener("submit", function(event) {
    event.preventDefault(); // 阻止表单默认提交行为

    // 获取表单数据
    const age = document.getElementById("age").value;
    const gender = document.getElementById("gender").value;
    const dob = document.getElementById("dob").value;
    const medicalRecords = document.getElementById("medicalRecords").value;
    const imageUpload = document.getElementById("imageUpload").files[0];

    // 简单的表单验证（这里只是示例，实际项目中可能需要更复杂的验证逻辑）
    if (!age || !gender || !dob || !medicalRecords || !imageUpload) {
        alert("请填写所有必填项");
        return;
    }

    // 创建一个 FormData 对象来存储表单数据
    const formData = new FormData();
    formData.append("age", age);
    formData.append("gender", gender);
    formData.append("dob", dob);
    formData.append("medicalRecords", medicalRecords);
    formData.append("imageUpload", imageUpload);
});