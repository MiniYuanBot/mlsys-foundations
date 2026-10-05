import unittest
from lab4.trace.workload import make_trace
from lab4.trace.audit import audit


class ReplayAudit(unittest.TestCase):
    def test_all_requests_and_send_lag(self):
        trace=make_trace(11,4,2,0.5)
        observations=[dict(request_id=r["request_id"],scheduled_s=r["arrival_s"],
                           actual_send_s=r["arrival_s"]+0.01,completed=i!=1,
                           error="timeout" if i==1 else None)
                      for i,r in enumerate(trace["requests"])]
        result=audit(trace,observations)
        self.assertEqual((result["num_requested"],result["num_completed"],result["num_failed"]),(4,3,1))
        self.assertEqual(result["missing_ids"],[])
        self.assertEqual(result["duplicate_ids"],[])
        self.assertAlmostEqual(result["max_send_lag_s"],0.01)
        self.assertTrue(result["arrival_valid"])
        observations[0]["actual_send_s"]+=1
        self.assertFalse(audit(trace,observations)["arrival_valid"])

    def test_missing_duplicate_and_early(self):
        trace=make_trace(22,3,1,0)
        rows=[dict(request_id=r["request_id"],scheduled_s=r["arrival_s"],
                   actual_send_s=r["arrival_s"],completed=True,error=None) for r in trace["requests"]]
        self.assertFalse(audit(trace,rows[:-1])["arrival_valid"])
        self.assertFalse(audit(trace,rows+[rows[0]])["arrival_valid"])
        rows[0]["actual_send_s"]-=0.1
        self.assertFalse(audit(trace,rows)["arrival_valid"])

