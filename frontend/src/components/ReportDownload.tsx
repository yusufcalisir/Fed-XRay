"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { FileText, Download } from "lucide-react";
import { getPdfReportUrl } from "@/lib/api";

interface ReportDownloadProps { diagnosisName: string; confidence: number; }

export default function ReportDownload({ diagnosisName, confidence }: ReportDownloadProps) {
  const { t } = useLanguage();
  const { apiUrl } = useApi();
  const pdfUrl = getPdfReportUrl(apiUrl, diagnosisName, confidence);

  return (
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-1">
        <FileText className="w-4 h-4 text-red-400" />
        <span className="text-sm font-bold text-white">{t("sec3_report_title")}</span>
      </div>
      <p className="text-[11px] text-slate-500 mb-4">{t("sec3_report_desc")}</p>
      <a href={pdfUrl} target="_blank" rel="noopener noreferrer" className="btn-primary w-full !bg-gradient-to-r !from-red-600 !to-red-500 hover:!from-red-500 hover:!to-red-400 !shadow-red-500/20">
        <Download className="w-4 h-4" />
        <span>{t("sec3_btn_download_pdf")}</span>
      </a>
    </div>
  );
}
