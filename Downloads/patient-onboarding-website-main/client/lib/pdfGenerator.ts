import { UserData, OnboardingProgress, AppointmentData } from "./UserContext";

// PDF generation utility without external dependencies
export interface PDFData {
  userData: UserData;
  onboardingProgress: OnboardingProgress;
  appointments: AppointmentData[];
  completionDate: string;
}

export const generateOnboardingReceiptPDF = (data: PDFData): void => {
  // Create a comprehensive HTML structure for PDF conversion
  const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>HealthCare Plus - Onboarding Receipt</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            text-align: center;
            border-bottom: 3px solid #0284c7;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .logo {
            font-size: 28px;
            font-weight: bold;
            color: #0284c7;
            margin-bottom: 5px;
        }
        .subtitle {
            color: #666;
            font-size: 16px;
        }
        .section {
            margin-bottom: 25px;
            padding: 15px;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
        }
        .section-title {
            font-size: 18px;
            font-weight: bold;
            color: #0284c7;
            margin-bottom: 15px;
            border-bottom: 1px solid #e5e7eb;
            padding-bottom: 5px;
        }
        .info-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
        }
        .info-item {
            margin-bottom: 10px;
        }
        .info-label {
            font-weight: bold;
            color: #374151;
            margin-bottom: 2px;
        }
        .info-value {
            color: #666;
        }
        .status-complete {
            color: #059669;
            font-weight: bold;
        }
        .status-incomplete {
            color: #dc2626;
        }
        .progress-bar {
            width: 100%;
            height: 20px;
            background-color: #e5e7eb;
            border-radius: 10px;
            overflow: hidden;
            margin: 10px 0;
        }
        .progress-fill {
            height: 100%;
            background-color: #059669;
            transition: width 0.3s ease;
        }
        .appointment-card {
            border: 1px solid #d1d5db;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 10px;
            background-color: #f9fafb;
        }
        .footer {
            text-align: center;
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #e5e7eb;
            color: #666;
            font-size: 14px;
        }
        @media print {
            body { margin: 0; }
            .section { break-inside: avoid; }
        }
    </style>
</head>
<body>
    <div class="header">
        <div class="logo">HealthCare Plus</div>
        <div class="subtitle">Patient Onboarding Receipt</div>
        <div class="subtitle">Generated on ${new Date().toLocaleDateString()}</div>
    </div>

    <div class="section">
        <div class="section-title">Patient Information</div>
        <div class="info-grid">
            <div class="info-item">
                <div class="info-label">Full Name:</div>
                <div class="info-value">${data.userData.firstName} ${data.userData.lastName}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Email:</div>
                <div class="info-value">${data.userData.email}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Phone:</div>
                <div class="info-value">${data.userData.phone}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Date of Birth:</div>
                <div class="info-value">${new Date(data.userData.dateOfBirth).toLocaleDateString()}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Gender:</div>
                <div class="info-value">${data.userData.gender.charAt(0).toUpperCase() + data.userData.gender.slice(1)}</div>
            </div>
        </div>
    </div>

    <div class="section">
        <div class="section-title">Onboarding Progress</div>
        ${(() => {
          const tasks = [
            { key: "uploadDocuments", name: "Document Upload & Verification" },
            { key: "scheduleAppointment", name: "First Appointment Scheduled" },
            { key: "medicalHistory", name: "Medical History Completed" },
            { key: "emergencyContacts", name: "Emergency Contacts Added" },
            { key: "consentForms", name: "Consent Forms Signed" },
          ];
          const completedTasks = tasks.filter(
            (task) =>
              data.onboardingProgress[task.key as keyof OnboardingProgress],
          ).length;
          const totalTasks = tasks.length;
          const percentage = Math.round((completedTasks / totalTasks) * 100);

          return `
            <div style="margin-bottom: 15px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                    <span><strong>Overall Progress:</strong></span>
                    <span><strong>${completedTasks}/${totalTasks} tasks completed (${percentage}%)</strong></span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
            ${tasks
              .map(
                (task) => `
                <div class="info-item">
                    <div class="info-label">${task.name}:</div>
                    <div class="info-value ${data.onboardingProgress[task.key as keyof OnboardingProgress] ? "status-complete" : "status-incomplete"}">
                        ${data.onboardingProgress[task.key as keyof OnboardingProgress] ? "Completed" : "Pending"}
                    </div>
                </div>
            `,
              )
              .join("")}
          `;
        })()}
    </div>

    ${
      data.appointments.length > 0
        ? `
    <div class="section">
        <div class="section-title">Scheduled Appointments</div>
        ${data.appointments
          .map(
            (apt) => `
            <div class="appointment-card">
                <div style="display: flex; justify-content: space-between; align-items: start; margin-bottom: 8px;">
                    <div>
                        <div class="info-label">${apt.appointmentType}</div>
                        <div class="info-value">with ${apt.doctorName}</div>
                    </div>
                    <div class="status-${apt.status === "scheduled" ? "complete" : "incomplete"}" style="font-size: 12px; padding: 2px 8px; border-radius: 12px; background-color: ${apt.status === "scheduled" ? "#dcfce7" : "#fee2e2"};">
                        ${apt.status.toUpperCase()}
                    </div>
                </div>
                <div class="info-grid">
                    <div class="info-item">
                        <div class="info-label">Date:</div>
                        <div class="info-value">${new Date(apt.date).toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Time:</div>
                        <div class="info-value">${apt.time}</div>
                    </div>
                    <div class="info-item">
                        <div class="info-label">Location:</div>
                        <div class="info-value">${apt.location}</div>
                    </div>
                </div>
            </div>
        `,
          )
          .join("")}
    </div>
    `
        : ""
    }

    <div class="section">
        <div class="section-title">Important Information</div>
        <div class="info-item">
            <div class="info-label">Registration Completed:</div>
            <div class="info-value">${data.completionDate}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Patient ID:</div>
            <div class="info-value">HC-${data.userData.email.split("@")[0].toUpperCase()}-${Date.now().toString().slice(-6)}</div>
        </div>
        <div class="info-item">
            <div class="info-label">Healthcare Provider:</div>
            <div class="info-value">HealthCare Plus Medical Center</div>
        </div>
        <div class="info-item">
            <div class="info-label">Contact Information:</div>
            <div class="info-value">
                Phone: 98765 43210<br>
                Email: support@healthcareplus.com<br>
                Address: Alpha Tower, 4th Floor, Bangalore
            </div>
        </div>
    </div>

    <div class="footer">
        <p><strong>HealthCare Plus</strong> - Your Health, Our Priority</p>
        <p>This document contains confidential patient information protected under HIPAA regulations.</p>
        <p>Generated automatically on ${new Date().toLocaleString()}</p>
    </div>
</body>
</html>`;

  // Create a new window/tab with the HTML content
  const printWindow = window.open("", "_blank");
  if (printWindow) {
    printWindow.document.write(htmlContent);
    printWindow.document.close();

    // Wait for content to load, then trigger print dialog
    setTimeout(() => {
      printWindow.print();

      // Optional: Close the window after printing (uncomment if desired)
      // setTimeout(() => printWindow.close(), 1000);
    }, 500);
  } else {
    // Fallback: Download as HTML file if popup is blocked
    downloadHTMLFile(
      htmlContent,
      `HealthCarePlus_Receipt_${data.userData.firstName}_${data.userData.lastName}.html`,
    );
  }
};

// Fallback function to download as HTML file
const downloadHTMLFile = (content: string, filename: string): void => {
  const blob = new Blob([content], { type: "text/html" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};

// Alternative: Generate and download as a comprehensive text file
export const generateTextReceipt = (data: PDFData): void => {
  const tasks = [
    { key: "uploadDocuments", name: "Document Upload & Verification" },
    { key: "scheduleAppointment", name: "First Appointment Scheduled" },
    { key: "medicalHistory", name: "Medical History Completed" },
    { key: "emergencyContacts", name: "Emergency Contacts Added" },
    { key: "consentForms", name: "Consent Forms Signed" },
  ];
  const completedTasks = tasks.filter(
    (task) => data.onboardingProgress[task.key as keyof OnboardingProgress],
  ).length;
  const totalTasks = tasks.length;
  const percentage = Math.round((completedTasks / totalTasks) * 100);

  let progress = `Overall Progress: ${completedTasks}/${totalTasks} tasks completed (${percentage}%)\n\n`;
  tasks.forEach((task) => {
    const status = data.onboardingProgress[task.key as keyof OnboardingProgress]
      ? "COMPLETED"
      : "PENDING";
    progress += `• ${task.name}: ${status}\n`;
  });

  const textContent = `
HEALTHCARE PLUS - PATIENT ONBOARDING RECEIPT
===========================================

Generated: ${new Date().toLocaleString()}

PATIENT INFORMATION
-------------------
Name: ${data.userData.firstName} ${data.userData.lastName}
Email: ${data.userData.email}
Phone: ${data.userData.phone}
Date of Birth: ${new Date(data.userData.dateOfBirth).toLocaleDateString()}
Gender: ${data.userData.gender.charAt(0).toUpperCase() + data.userData.gender.slice(1)}

ONBOARDING PROGRESS
-------------------
${progress}

${
  data.appointments.length > 0
    ? `
SCHEDULED APPOINTMENTS
---------------------
${data.appointments
  .map(
    (apt) => `
• ${apt.appointmentType} with ${apt.doctorName}
  Date: ${new Date(apt.date).toLocaleDateString("en-US", { weekday: "long", year: "numeric", month: "long", day: "numeric" })}
  Time: ${apt.time}
  Location: ${apt.location}
  Status: ${apt.status.toUpperCase()}
`,
  )
  .join("")}`
    : ""
}

IMPORTANT INFORMATION
--------------------
Registration Completed: ${data.completionDate}
Patient ID: HC-${data.userData.email.split("@")[0].toUpperCase()}-${Date.now().toString().slice(-6)}
Healthcare Provider: HealthCare Plus Medical Center

CONTACT INFORMATION
------------------
Phone: 98765 43210
Email: support@healthcareplus.com
Address: Alpha Tower, 4th Floor, Bangalore

===========================================
HealthCare Plus - Your Health, Our Priority
This document contains confidential patient information protected under HIPAA regulations.
Generated automatically on ${new Date().toLocaleString()}
`;

  const blob = new Blob([textContent], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `HealthCarePlus_Receipt_${data.userData.firstName}_${data.userData.lastName}.txt`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(url);
};
