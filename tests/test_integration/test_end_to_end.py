"""
End-to-End Integration Tests for ResumeAI
Tests the complete pipeline from input to output.
"""

import sys
from pathlib import Path
import pandas as pd

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.pipeline import ResumePipeline, rank_candidates
import config


def test_config_weights():
    """Test that configuration weights are correct."""
    print("=" * 70)
    print("TEST 1: Configuration Weights")
    print("=" * 70)

    weights = config.WEIGHTS

    print(f"\nCurrent weights:")
    print(f"  Skills: {weights['skills']}")
    print(f"  Experience: {weights['experience']}")
    print(f"  Semantic: {weights['semantic']}")
    print(f"  Education: {weights['education']}")
    print(f"  Location: {weights['location']}")

    # Check weights sum to 1.0
    total = sum(weights.values())
    print(f"\n  Total: {total}")

    assert 0.99 <= total <= 1.01, f"Weights should sum to 1.0, got {total}"

    # Check location is 0
    assert weights['location'] == 0.0, f"Location weight should be 0, got {weights['location']}"

    print("\n✓ Configuration test PASSED")
    print("=" * 70)


def test_pipeline_initialization():
    """Test that pipeline initializes correctly."""
    print("\n" + "=" * 70)
    print("TEST 2: Pipeline Initialization")
    print("=" * 70)

    try:
        pipeline = ResumePipeline()
        print("\n✓ Pipeline initialized successfully")
        print(f"  Location scoring enabled: {pipeline.use_location}")
        print(f"  Scoring modules: {list(pipeline.scoring_modules.keys())}")
        print("\n✓ Pipeline initialization test PASSED")
    except Exception as e:
        print(f"\n✗ Pipeline initialization FAILED: {e}")
        raise

    print("=" * 70)


def test_end_to_end_ranking():
    """Test end-to-end ranking with sample data."""
    print("\n" + "=" * 70)
    print("TEST 3: End-to-End Ranking")
    print("=" * 70)

    # Get sample files
    resume_file = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    job_file = project_root / "tests" / "test_job_parser" / "sample_job_description.txt"

    if not resume_file.exists():
        print(f"\n⚠️  Warning: Resume file not found at {resume_file}")
        print("Skipping test...")
        return

    if not job_file.exists():
        print(f"\n⚠️  Warning: Job file not found at {job_file}")
        print("Skipping test...")
        return

    print(f"\nTest files:")
    print(f"  Resume: {resume_file.name}")
    print(f"  Job: {job_file.name}")

    # Read job description
    with open(job_file, 'r') as f:
        job_text = f.read()

    try:
        # Run pipeline (without saving)
        pipeline = ResumePipeline()
        results = pipeline.run(
            job_text=job_text,
            resume_files=[str(resume_file)],
            save_output=False,
            return_format='dataframe'
        )

        # Validate results
        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        assert len(results) > 0, "Results should not be empty"
        assert 'final_score' in results.columns, "Results should have final_score column"

        # Check score components
        required_columns = ['rank', 'name', 'email', 'final_score',
                           'skills_score', 'experience_score', 'semantic_score',
                           'education_score', 'location_score']

        for col in required_columns:
            assert col in results.columns, f"Results should have {col} column"

        # Check scores are in valid range
        for col in ['final_score', 'skills_score', 'experience_score',
                   'semantic_score', 'education_score']:
            scores = results[col]
            assert scores.between(0, 1).all(), f"{col} should be between 0 and 1"

        # Print results
        print(f"\n✓ End-to-end ranking successful!")
        print(f"\nResults:")
        print(f"  Candidate: {results.iloc[0]['name']}")
        print(f"  Final Score: {results.iloc[0]['final_score']:.3f}")
        print(f"  Skills: {results.iloc[0]['skills_score']:.3f}")
        print(f"  Experience: {results.iloc[0]['experience_score']:.3f}")
        print(f"  Semantic: {results.iloc[0]['semantic_score']:.3f}")
        print(f"  Education: {results.iloc[0]['education_score']:.3f}")
        print(f"  Location: {results.iloc[0]['location_score']:.3f}")

        print("\n✓ End-to-end ranking test PASSED")

    except Exception as e:
        print(f"\n✗ End-to-end ranking FAILED: {e}")
        raise

    print("=" * 70)


def test_convenience_function():
    """Test the rank_candidates convenience function."""
    print("\n" + "=" * 70)
    print("TEST 4: Convenience Function")
    print("=" * 70)

    # Get sample files
    resume_file = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    job_file = project_root / "tests" / "test_job_parser" / "sample_job_description.txt"

    if not resume_file.exists() or not job_file.exists():
        print("\n⚠️  Warning: Test files not found")
        print("Skipping test...")
        return

    # Read job description
    with open(job_file, 'r') as f:
        job_text = f.read()

    try:
        # Test convenience function
        results = rank_candidates(
            job_text=job_text,
            resume_files=[str(resume_file)],
            save_output=False
        )

        assert isinstance(results, pd.DataFrame), "Results should be a DataFrame"
        assert len(results) > 0, "Results should not be empty"

        print(f"\n✓ Convenience function works!")
        print(f"  Results: {len(results)} candidate(s) ranked")

        print("\n✓ Convenience function test PASSED")

    except Exception as e:
        print(f"\n✗ Convenience function FAILED: {e}")
        raise

    print("=" * 70)


def test_scoring_weights_applied():
    """Test that scoring weights are correctly applied."""
    print("\n" + "=" * 70)
    print("TEST 5: Scoring Weights Application")
    print("=" * 70)

    # Get sample files
    resume_file = project_root / "tests" / "test_parser" / "test_resume" / "Shivendra_Resume.docx"
    job_file = project_root / "tests" / "test_job_parser" / "sample_job_description.txt"

    if not resume_file.exists() or not job_file.exists():
        print("\n⚠️  Warning: Test files not found")
        print("Skipping test...")
        return

    # Read job description
    with open(job_file, 'r') as f:
        job_text = f.read()

    try:
        results = rank_candidates(
            job_text=job_text,
            resume_files=[str(resume_file)],
            save_output=False
        )

        # Calculate expected final score manually
        row = results.iloc[0]
        expected_score = (
            config.WEIGHTS['skills'] * row['skills_score'] +
            config.WEIGHTS['experience'] * row['experience_score'] +
            config.WEIGHTS['semantic'] * row['semantic_score'] +
            config.WEIGHTS['education'] * row['education_score'] +
            config.WEIGHTS['location'] * row['location_score']
        )

        actual_score = row['final_score']

        # Allow small floating point differences
        diff = abs(expected_score - actual_score)
        assert diff < 0.001, f"Final score mismatch: expected {expected_score:.4f}, got {actual_score:.4f}"

        print(f"\n✓ Weights correctly applied!")
        print(f"  Expected: {expected_score:.4f}")
        print(f"  Actual: {actual_score:.4f}")
        print(f"  Difference: {diff:.6f}")

        # Verify location contributes 0
        location_contribution = config.WEIGHTS['location'] * row['location_score']
        assert location_contribution == 0.0, f"Location should contribute 0, got {location_contribution}"

        print(f"\n✓ Location contribution: {location_contribution} (correct)")

        print("\n✓ Scoring weights test PASSED")

    except Exception as e:
        print(f"\n✗ Scoring weights test FAILED: {e}")
        raise

    print("=" * 70)


def run_all_tests():
    """Run all integration tests."""
    print("\n" + "=" * 70)
    print("RESUMEAI - END-TO-END INTEGRATION TESTS")
    print("=" * 70)
    print()

    tests = [
        test_config_weights,
        test_pipeline_initialization,
        test_end_to_end_ranking,
        test_convenience_function,
        test_scoring_weights_applied
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"\n❌ Test failed: {test.__name__}")
            print(f"   Error: {e}")
            failed += 1
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    print(f"  Total tests: {len(tests)}")
    print(f"  ✓ Passed: {passed}")
    print(f"  ✗ Failed: {failed}")

    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
    else:
        print(f"\n⚠️  {failed} test(s) failed")

    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
